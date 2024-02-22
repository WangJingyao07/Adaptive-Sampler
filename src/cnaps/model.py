from src.cnaps.adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
from src.cnaps.resnet import film_resnet18, resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

NUM_SAMPLES = 1


def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


class NormalizationLayer(nn.BatchNorm2d):
    """
    Base class for all normalization layers.
    Derives from nn.BatchNorm2d to maintain compatibility with the pre-trained resnet-18.
    """

    def __init__(self, num_features):
        """
        Initialize the class.
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(NormalizationLayer, self).__init__(
            num_features,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True)

    def forward(self, x):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        pass  # always override this method

    def _normalize(self, x, mean, var):
        """
        Normalize activations.
        :param x: input activations
        :param mean: mean used to normalize
        :param var: var used to normalize
        :return: normalized activations
        """
        return (self.weight.view(1, -1, 1, 1) * (x - mean) / torch.sqrt(var + self.eps)) + self.bias.view(1, -1, 1, 1)

    @staticmethod
    def _compute_batch_moments(x):
        """
        Compute conventional batch mean and variance.
        :param x: input activations
        :return: batch mean, batch variance
        """
        return torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(x, dim=(0, 2, 3), keepdim=True)

    @staticmethod
    def _compute_instance_moments(x):
        """
        Compute instance mean and variance.
        :param x: input activations
        :return: instance mean, instance variance
        """
        return torch.mean(x, dim=(2, 3), keepdim=True), torch.var(x, dim=(2, 3), keepdim=True)

    @staticmethod
    def _compute_layer_moments(x):
        """
        Compute layer mean and variance.
        :param x: input activations
        :return: layer mean, layer variance
        """
        return torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)

    @staticmethod
    def _compute_pooled_moments(x, alpha, batch_mean, batch_var, augment_moment_fn):
        """
        Combine batch moments with augment moments using blend factor alpha.
        :param x: input activations
        :param alpha: moment blend factor
        :param batch_mean: standard batch mean
        :param batch_var: standard batch variance
        :param augment_moment_fn: function to compute augment moments
        :return: pooled mean, pooled variance
        """
        augment_mean, augment_var = augment_moment_fn(x)
        pooled_mean = alpha * batch_mean + (1.0 - alpha) * augment_mean
        batch_mean_diff = batch_mean - pooled_mean
        augment_mean_diff = augment_mean - pooled_mean
        pooled_var = alpha * (batch_var + (batch_mean_diff * batch_mean_diff)) +\
            (1.0 - alpha) * (augment_var + (augment_mean_diff * augment_mean_diff))
        return pooled_mean, pooled_var


class TaskNormBase(NormalizationLayer):
    """TaskNorm base class."""

    def __init__(self, num_features):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNormBase, self).__init__(num_features)
        self.sigmoid = torch.nn.Sigmoid()

    def register_extra_weights(self):
        """
        The parameters here get registered after initialization because the pre-trained resnet model does not have
        these parameters and would fail to load if these were declared at initialization.
        :return: Nothing
        """
        device = self.weight.device

        # Initialize and register the learned parameters 'a' (SCALE) and 'b' (OFFSET)
        # for calculating alpha as a function of context size.
        a = torch.Tensor([0.0]).to(device)
        b = torch.Tensor([0.0]).to(device)
        self.register_parameter(name='a', param=torch.nn.Parameter(a, requires_grad=True))
        self.register_parameter(name='b', param=torch.nn.Parameter(b, requires_grad=True))

        # Variables to store the context moments to use for normalizing the target.
        self.register_buffer(name='batch_mean',
                             tensor=torch.zeros((1, self.num_features, 1, 1), requires_grad=True, device=device))
        self.register_buffer(name='batch_var',
                             tensor=torch.ones((1, self.num_features, 1, 1), requires_grad=True, device=device))

        # Variable to save the context size.
        self.register_buffer(name='context_size',
                             tensor=torch.zeros((1), requires_grad=False, device=device))

    def _get_augment_moment_fn(self):
        """
        Provides the function to compute augment moemnts.
        :return: function to compute augment moments.
        """
        pass  # always override this function

    def forward(self, x):
        """
        Normalize activations.
        :param x: input activations
        :return: normalized activations
        """
        if self.training:  # compute the pooled moments for the context and save off the moments and context size
            alpha = self.sigmoid(self.a * (x.size())[0] + self.b)  # compute alpha with context size
            batch_mean, batch_var = self._compute_batch_moments(x)
            pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, batch_mean, batch_var,
                                                                   self._get_augment_moment_fn())
            self.context_batch_mean = batch_mean
            self.context_batch_var = batch_var
            self.context_size = torch.full_like(self.context_size, x.size()[0])
        else:  # compute the pooled moments for the target
            # compute alpha with saved context size
            alpha = self.sigmoid(self.a * self.context_size + self.b)
            pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, self.context_batch_mean,
                                                                   self.context_batch_var,
                                                                   self._get_augment_moment_fn())

        return self._normalize(x, pooled_mean, pooled_var)  # normalize


class TaskNormI(TaskNormBase):
    """
    TaskNorm-I normalization layer. Just need to override the augment moment function with 'instance'.
    """

    def __init__(self, num_features):
        """
        Initialize
        :param num_features: number of channels in the 2D convolutional layer
        """
        super(TaskNormI, self).__init__(num_features)

    def _get_augment_moment_fn(self):
        """
        Override the base class to get the function to compute instance moments.
        :return: function to compute instance moments
        """
        return self._compute_instance_moments


class SimplePrePoolNet(nn.Module):
    """
    Simple prepooling network for images. Implements the phi mapping in DeepSets networks. In this work we use a
    multi-layer convolutional network similar to that in https://openreview.net/pdf?id=rJY0-Kcll.
    """

    def __init__(self, batch_normalization):
        super(SimplePrePoolNet, self).__init__()
        if batch_normalization == "task_norm-i":
            self.layer1 = self._make_conv2d_layer_task_norm(3, 64)
            self.layer2 = self._make_conv2d_layer_task_norm(64, 64)
            self.layer3 = self._make_conv2d_layer_task_norm(64, 64)
            self.layer4 = self._make_conv2d_layer_task_norm(64, 64)
            self.layer5 = self._make_conv2d_layer_task_norm(64, 64)
        else:
            self.layer1 = self._make_conv2d_layer(3, 64)
            self.layer2 = self._make_conv2d_layer(64, 64)
            self.layer3 = self._make_conv2d_layer(64, 64)
            self.layer4 = self._make_conv2d_layer(64, 64)
            self.layer5 = self._make_conv2d_layer(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _make_conv2d_layer(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    @staticmethod
    def _make_conv2d_layer_task_norm(in_maps, out_maps):
        return nn.Sequential(
            nn.Conv2d(in_maps, out_maps, kernel_size=3, stride=1, padding=1),
            TaskNormI(out_maps),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    @property
    def output_size(self):
        return 64


class SetEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """

    def __init__(self, batch_normalization):
        super(SetEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet(batch_normalization)
        self.pooling_fn = mean_pooling

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:
                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.
        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = self.pre_pooling_fn(x)
        x = self.pooling_fn(x)
        return x


class ConfigureNetworks:
    """ Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
    """

    def __init__(self, pretrained_resnet_path, feature_adaptation, batch_normalization):
        self.classifier = linear_classifier

        self.encoder = SetEncoder(batch_normalization)
        z_g_dim = self.encoder.pre_pooling_fn.output_size

        # parameters for ResNet18
        num_maps_per_layer = [64, 128, 256, 512]
        num_blocks_per_layer = [2, 2, 2, 2]
        num_initial_conv_maps = 64

        if feature_adaptation == "no_adaptation":
            self.feature_extractor = resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                batch_normalization=batch_normalization
            )
            self.feature_adaptation_network = NullFeatureAdaptationNetwork()

        elif feature_adaptation == "film":
            self.feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                batch_normalization=batch_normalization
            )
            self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                z_g_dim=z_g_dim
            )

        elif feature_adaptation == 'film+ar':
            self.feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                batch_normalization=batch_normalization
            )
            self.feature_adaptation_network = FilmArAdaptationNetwork(
                feature_extractor=self.feature_extractor,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                num_initial_conv_maps=num_initial_conv_maps,
                z_g_dim=z_g_dim
            )

        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(
            self.feature_extractor.output_size)

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier

    def get_classifier_adaptation(self):
        return self.classifier_adaptation_network

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


class Cnaps(nn.Module):
    """
    Main model class. Implements several CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """

    def __init__(self,
                 pretrained_resnet_path='./src/cnaps/checkpoints/pretrained_resnet.pt.tar',
                 feature_adaptation='film',
                 batch_normalization='tasknorm-i'):
        super(Cnaps, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        networks = ConfigureNetworks(pretrained_resnet_path=pretrained_resnet_path,
                                     feature_adaptation=feature_adaptation,
                                     batch_normalization=batch_normalization)

        self.pretrained_resnet_path = pretrained_resnet_path
        self.feature_adaptation = feature_adaptation
        self.batch_normalization = batch_normalization
        self.set_encoder = networks.get_encoder()
        self.classifier_adaptation_network = networks.get_classifier_adaptation()
        self.classifier = networks.get_classifier()
        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.use_two_gpus = True
        # Dictionary mapping class label (integer) to encoded representation
        self.class_representations = OrderedDict()

    def forward(self, context_images, context_labels, target_images):
        """
        Forward pass through the model for one episode.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param context_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """
        if context_images.size()[1] == 1:
            context_images = context_images.repeat(1, 3, 1, 1)
            target_images = target_images.repeat(1, 3, 1, 1)
        # extract train and test features
        self.task_representation = self.set_encoder(context_images)
        context_features, target_features = self._get_features(context_images, target_images)

        # get the parameters for the linear classifier.
        self._build_class_reps(context_features, context_labels)
        classifier_params = self._get_classifier_params()

        # classify
        sample_logits = self.classifier(target_features, classifier_params)
        self.class_representations.clear()

        # this adds back extra first dimension for num_samples
        return split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_images.shape[0]]), context_features

    def _get_features(self, context_images, target_images):
        """
        Helper function to extract task-dependent feature representation for each image in both context and target sets.
        :param context_images: (torch.tensor) Images in the context set (batch x C x H x W).
        :param target_images: (torch.tensor) Images in the target set (batch x C x H x W).
        :return: (tuple::torch.tensor) Feature representation for each set of images.
        """
        # Parallelize forward pass across multiple GPUs (model parallelism)
        if self.use_two_gpus:
            context_images_1 = context_images.cuda(1)
            target_images_1 = target_images.cuda(1)
            self.feature_adaptation_network.to(device='cuda:1')
            self.feature_extractor.to(device='cuda:1')
            if self.feature_adaptation == 'film+ar':
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.set_batch_norm_mode(True)
                self.feature_extractor_params = self.feature_adaptation_network(
                    context_images_1, task_representation_1)
            else:
                task_representation_1 = self.task_representation.cuda(1)
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(
                    task_representation_1)
            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            self.set_batch_norm_mode(True)
            context_features_1 = self.feature_extractor(
                context_images_1, self.feature_extractor_params)
            context_features = context_features_1.cuda(0)
            self.set_batch_norm_mode(False)
            target_features_1 = self.feature_extractor(
                target_images_1, self.feature_extractor_params)
            target_features = target_features_1.cuda(0)
        else:
            if self.feature_adaptation == 'film+ar':
                # Get adaptation params by passing context set through the adaptation networks
                self.set_batch_norm_mode(True)
                self.feature_extractor_params = self.feature_adaptation_network(
                    context_images, self.task_representation)
            else:
                # Get adaptation params by passing context set through the adaptation networks
                self.feature_extractor_params = self.feature_adaptation_network(
                    self.task_representation)
            # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
            self.set_batch_norm_mode(True)
            context_features = self.feature_extractor(context_images, self.feature_extractor_params)
            self.set_batch_norm_mode(False)
            target_features = self.feature_extractor(target_images, self.feature_extractor_params)

        return context_features, target_features

    def _build_class_reps(self, context_features, context_labels):
        """
        Construct and return class level representation for each class in task.
        :param context_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation dictionary.
        """
        for c in torch.unique(context_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(
                context_features, 0, self._extract_class_indices(context_labels, c))
            class_rep = mean_pooling(class_features)
            self.class_representations[c.item()] = class_rep

    def _get_classifier_params(self):
        """
        Processes the class representations and generated the linear classifier weights and biases.
        :return: Linear classifier weights and biases.
        """
        classifier_params = self.classifier_adaptation_network(self.class_representations)
        return classifier_params

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        # indices of labels equal to which class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        """
        Moves the feature extractor and feature adaptation network to a second GPU.
        :return: Nothing
        """
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)

    def set_batch_norm_mode(self, context):
        """
        Controls the batch norm mode in the feature extractor.
        :param context: Set to true ehen processing the context set and False when processing the target set.
        :return: Nothing
        """
        if self.batch_normalization == "basic":
            self.feature_extractor.eval()  # always in eval mode
        else:
            # "task_norm-i" - respect context flag, regardless of state
            if context:
                self.feature_extractor.train()  # use train when processing the context set
            else:
                self.feature_extractor.eval()  # use eval when processing the target set
