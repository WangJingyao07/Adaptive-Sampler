from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset

from torchmeta.utils.data.dataset import CombinationMetaDataset
import random
import warnings
from torch.utils.data.sampler import RandomSampler
import ast
from itertools import combinations
from itertools import chain


class TrainWithASr(RandomSampler):
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        self.asr_weights = self.initialize_asr_weights()
        if FLAGS.datasource == 'sinusoid':
            self.dim_hidden = [40, 40]
            self.loss_func = mse
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'cifarfs':
            self.loss_func = xent
            self.classification = True

            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward = self.forward_fc
                self.construct_weights = self.construct_fc_weights

            # Determine amount of channels to use
            if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'cifarfs':
                self.channels = 3
            else:
                self.channels = 1

            # Compute image width (=height)
            self.img_size = int(
                np.sqrt(self.dim_input / self.channels))  # dim input is length of totally flattened image
        else:
            raise ValueError('Unrecognized data source.')

    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        num_classes = len(self.data_source.dataset)
        if len(self.tasks):
            x = self.tasks
            for _ in range(len(self.tasks)):
                y = random.sample(x, 1)
                x = [item for item in x if item not in y]
                yield tuple(y[0])
            for _ in range(int(self.batch_size-len(self.tasks))):
                yield tuple(random.sample(range(num_classes), num_classes_per_task))
        else:
            for _ in range(self.batch_size):
                yield tuple(random.sample(range(num_classes), num_classes_per_task))

    # Define ASr Task Diversity Calculation
    def calculate_task_diversity(self, inputa, labela):
        # Calculate Z_i and other necessary variables
        Z_i = encode_task(inputa)  # Replace with your task encoding function
        n = len(inputa)  # Number of samples in task
        d = Z_i.shape[1]  # Dimension of Z_i

        # Parameters
        sigma = 0.1  # Adjust as needed

        I = torch.eye(d)
        Z_i_star = torch.transpose(Z_i, 0, 1)
        term = torch.mm(Z_i, Z_i_star) / (n * sigma ** 2)
        diversity = 0.5 * n * torch.log(torch.det(I + term))
        return diversity

    # Define ASr Task Entropy Calculation
    def calculate_task_entropy(self, inputa, labela):
        # Calculate C_i and other necessary variables
        C_i = calculate_C_i(inputa, labela)  # Replace with your class-specific calculations
        k = len(C_i)  # Number of classes in the task

        # Parameters
        epsilon = 0.1  # Adjust as needed

        I = torch.eye(d)
        entropy = 0

        for j in range(k):
            tr_C_i_j = torch.trace(C_i[j])
            term = Z_i @ C_i[j] @ torch.transpose(Z_i, 0, 1) / (tr_C_i_j * epsilon ** 2)
            entropy += (0.5 * tr_C_i_j / n) * torch.log(torch.det(I + term))

        return entropy

    # Define ASr Task Difficulty Calculation
    def calculate_task_difficulty(self, inputa, labela):
        # Calculate gradients using the model
        loss = self.loss_func(self.forward(inputa, weights, reuse=True), labela)
        gradients = torch.autograd.grad(loss, weights.values())

        difficulty = 0
        for grad in gradients:
            difficulty += torch.norm(grad, p=2) ** 2

        return difficulty

    def initialize_asr_weights(self):
        return tf.Variable(tf.ones(self.input_dim), name='asr_weights', trainable=True, dtype=tf.float32)

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # This function constructs the model, and defines the ops. The ops are not called yet! That happens in session.run(...)

        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:  # Directly couple input tensors from tf queue to object variables
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                # weights were already initialized during some training, reuse those
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                # this is done when construct_model is called
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates
            accuraciesb = [[]] * num_updates

            def task_metalearn(inp, reuse=True):
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                if self.classification:
                    task_accuraciesb = []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                # MAML line 5: evaluate grads on train set (a)
                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))

                # MAML line 6: compute updates (adapted parameters) with ASr importance
                # Calculate ASr values for task diversity, entropy, and difficulty
                t_diversity = calculate_task_diversity(inputa,
                                                       labela)
                t_entropy = calculate_task_entropy(inputa, labela)
                t_difficulty = calculate_task_difficulty(inputa,
                                                         labela)
                # Calculate ASr importance weights
                asr_weights = asr_sampler([t_diversity, t_entropy, t_difficulty])

                # Update the model weights using ASr-guided gradient updates
                fast_weights = dict(
                    zip(weights.keys(), [
                        weights[key] - self.update_lr * asr_weights[key] * gradients[key]
                        for key in weights.keys()
                    ])
                )

                # Continue with the rest of the MAML training loop...

                # MAML line 8: calculate output/loss on test set (b)
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)

                    # MAML line 5: evaluate grads on train set (a)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))

                    # MAML line 6: compute updates (adapted parameters)
                    fast_weights = dict(zip(fast_weights.keys(), [
                        weights[key] - self.update_lr * asr_weights[key] * gradients[key]
                        for key in fast_weights.keys()
                    ]))

                    # MAML line 8: calculate output/loss on test set (b)
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1),
                                                                 tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(
                            tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                                                        tf.argmax(labelb, 1))
                        )
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm is not 'None':  # to initialize batch norm variables
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:  # accuracies are also stored
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])
            # THE REAL LEARNING CONSTRUCTION OCCURS HERE
            # IMPORTANT: executes in parallel for ALL TASKS in batch I guess? The inputs are formatted in a special way to contain multiple tasks?
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j
                                                  in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:  # FLAGS.metatrain_iterations = how many times to execute
                # This is the meta optimizer
                optimizer = tf.train.AdamOptimizer(self.meta_lr)

                # Compute gradients after num_updates
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates - 1])

                # Gradients are clipped by [-10,10] to avoid explosion?
                if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'cifarfs':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]

                # update parameters
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                          for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(
                    FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 = [
                    tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

    ### Network construction functions
    ## not CNN
    # only used for sinusoid, and for non convolutional DNN on image datasets.
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    # only used for sinusoid, and for non convolutional DNN on image datasets.
    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]

    ## CNN
    # initialize and return weights for CNN
    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output],
                                            initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        elif FLAGS.datasource == 'cifarfs':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 2 * 2, self.dim_output],
                                            initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    # return output of input image, with weights given as argument!
    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'cifarfs':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        return tf.matmul(hidden4, weights['w5']) + weights['b5']


class MetaDatasetRandomSampler(ASrSampler):
    def __iter__(self):
        num_classes_per_task = self.data_source.num_classes_per_task
        if len(self.tasks):
            x = self.tasks
            for _ in range(len(self.tasks)):
                y = random.sample(x, 1)
                x = [item for item in x if item not in y]
                yield tuple(y[0])
            for _ in range(int(self.batch_size-len(self.tasks))):
                source = random.randrange(len(self.data_source.dataset.sources))
                num_classes = len(self.data_source.dataset._class_datasets[source])
                offset = self.data_source.dataset._cum_num_classes[source]
                indices = random.sample(range(num_classes), num_classes_per_task)
                yield tuple(index + offset for index in indices)
        else:

            for _ in range(self.batch_size):
                source = random.randrange(len(self.data_source.dataset.sources))
                num_classes = len(self.data_source.dataset._class_datasets[source])
                offset = self.data_source.dataset._cum_num_classes[source]
                indices = random.sample(range(num_classes), num_classes_per_task)
                yield tuple(index + offset for index in indices)


class MetaDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, task=None):
        if collate_fn is None:
            collate_fn = no_collate

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            sampler = OHTMSampler(dataset, task, batch_size)
        shuffle = False

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
                                             shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
                                             num_workers=num_workers, collate_fn=collate_fn,
                                             pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class BatchMetaCollate(object):

    def __init__(self, collate_fn):
        super().__init__()
        self.collate_fn = collate_fn

    def collate_task(self, task):
        if isinstance(task, TorchDataset):
            return self.collate_fn([task[idx] for idx in range(len(task))])
        elif isinstance(task, OrderedDict):
            return OrderedDict([(key, self.collate_task(subtask))
                                for (key, subtask) in task.items()])
        else:
            raise NotImplementedError()

    def __call__(self, batch):
        return self.collate_fn([self.collate_task(task) for task in batch])


def no_collate(batch):
    return batch


class BatchMetaDataLoaderOHTM(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, task=None, use_batch_collate=True):
        if use_batch_collate:
            collate_fn = BatchMetaCollate(default_collate)
        else:
            collate_fn = default_collate
            sampler = MetaDatasetRandomSampler(dataset, task, batch_size)

        super(BatchMetaDataLoaderOHTM, self).__init__(dataset,
                                                      batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                                                      collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
                                                      timeout=timeout, worker_init_fn=worker_init_fn, task=task)