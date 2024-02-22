import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from src.cnaps.metalearners.loss import CNAPsLoss, aggregate_accuracy
import gc


class CNAPs(object):
    """Meta-learner class for Conditional Neural Adaptive Processes [1].
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.
    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).
    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.
    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].
    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.
    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates osqn the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.
    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].
    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.
    device : `torch.device` instance, optional
        The device on which the model is defined.
    References
    ----------
    .. [1] Requeima, James, et al. "Fast and flexible multi-task
           classification using conditional neural adaptive processes."
           Advances in Neural Information Processing Systems 32 (2019):
           7959-7970. (https://arxiv.org/pdf/1606.04080)
    """

    def __init__(self, model, optimizer=None, step_size=0.1,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=CNAPsLoss, device=None, num_ways=None,
                 num_shots=None, num_shots_test=None, ohtm=False, log_test_tasks=False):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test
        self.model.to(device=self.device)
        self.ohtm = ohtm
        self.log_test_tasks = log_test_tasks
        if self.ohtm:
            self.hardest_task = OrderedDict()
        if self.log_test_tasks:
            self.test_task_performance = OrderedDict()
        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                                                             dtype=param.dtype, device=self.device,
                                                             requires_grad=learn_step_size)) for (name, param)
                                         in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                                          device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                                            if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                                         for group in self.optimizer.param_groups])

    def get_loss(self, batch, train=False):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets, _ = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'loss': np.zeros((num_tasks,), dtype=np.float32),
            'mean_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies': np.zeros((num_tasks,), dtype=np.float32)
            })
        mean_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, task, test_inputs, test_targets, _) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            train_inputs = train_inputs.to(device=self.device)
            train_targets = train_targets.to(device=self.device)
            test_inputs = test_inputs.to(device=self.device)
            test_targets = test_targets.to(device=self.device)
            target_logits, _ = self.model(train_inputs, train_targets, test_inputs)
            loss = self.loss_function(target_logits, test_targets) / 16
            regularization_term = (
                self.model.feature_adaptation_network.regularization_term()).cuda(0)
            regularizer_scaling = 0.001
            loss += regularizer_scaling * regularization_term
            accuracy = aggregate_accuracy(target_logits, test_targets).detach().item()
            loss.backward(retain_graph=False)
            results['loss'][task_id] = loss.detach().item()
            mean_loss += loss.detach().item()

            if is_classification_task:
                results['accuracies'][task_id] = accuracy
            if self.ohtm and train:
                self.hardest_task[str(task.cpu().tolist())] = results['accuracies'][task_id]
            if self.log_test_tasks and not train:
                self.test_task_performance[str(task.cpu().tolist())
                                           ] = results['accuracies'][task_id]
            del train_inputs, train_targets, test_inputs, test_targets

        mean_loss.div_(num_tasks)
        results['mean_loss'] = mean_loss.item()
        return mean_loss, results

    def train(self, dataloader, max_batches=250, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_loss'])}
                if 'accuracies' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies']))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                               'optimizer is `None`. In order to train `{0}`, you must '
                               'specify a Pytorch optimizer as the argument of `{0}` '
                               '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                               'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                loss, results = self.get_loss(batch, train=True)
                self.optimizer.step()
                torch.cuda.empty_cache()
                gc.collect()
                yield results
                num_batches += 1

    def evaluate(self, dataloader, max_batches=200, verbose=True, **kwargs):
        mean_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_loss += (results['mean_loss']
                              - mean_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_loss)}
                if 'accuracies' in results:
                    mean_accuracy += (np.mean(results['accuracies'])
                                      - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_loss': mean_loss}
        if 'accuracies' in results:
            mean_results['accuracies'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if self.log_test_tasks:
                    if len(self.test_task_performance) == 1024:
                        break
                    else:
                        num_batches -= 1
                else:
                    if num_batches >= max_batches:
                        break
                _, results = self.get_loss(batch)
                yield results
                torch.cuda.empty_cache()
                gc.collect()
                num_batches += 1
            if self.log_test_tasks and len(self.test_task_performance) == 1024:
                break
