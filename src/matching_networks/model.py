import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchmeta.utils.matching import matching_log_probas, matching_loss


def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq


class Classifier(nn.Module):
    def __init__(self, layer_size=64, num_channels=1, keep_prob=1.0, image_size=28):
        super(Classifier, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = convLayer(layer_size, layer_size, keep_prob)

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Products pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarites of size[batch_size,sequence_length]
        :param support_set_y:[batch_size,sequence_length,classes_num]
        :return: Softmax pdf shape[batch_size,classes_num]
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(0).bmm(support_set_y).squeeze(0)
        return preds


class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """
        support_images_normed = F.normalize(support_set, p=2, dim=2)
        similarities = torch.bmm(support_images_normed, input_image.unsqueeze(0).permute(0, 2, 1))
        similarities = similarities.squeeze(dim=2)
        return similarities


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, batch_size, vector_dim, use_cuda):
        super(BidirectionalLSTM, self).__init__()
        """
        Initial a muti-layer Bidirectional LSTM
        :param layer_size: a list of each layer'size
        :param batch_size:
        :param vector_dim:
        """
        self.batch_size = batch_size
        self.hidden_size = layer_size[0]
        self.vector_dim = vector_dim
        self.num_layer = len(layer_size)
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,
                            bidirectional=True)
        self.hidden = self.init_hidden(self.use_cuda)

    def init_hidden(self, use_cuda):
        if use_cuda:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False).cuda(),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False).cuda())
        else:
            return (Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False),
                    Variable(torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size), requires_grad=False))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, inputs):
        # self.hidden = self.init_hidden(self.use_cuda)
        self.hidden = self.repackage_hidden(self.hidden)
        output, self.hidden = self.lstm(inputs, self.hidden)
        return output


class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob=1.0, batch_size=32, num_channels=1, fce=False, num_classes_per_set=5,
                 num_samples_per_class=1, image_size=28):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        self.fce = fce
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.image_size = image_size
        self.g = Classifier(layer_size=64, num_channels=num_channels,
                            keep_prob=keep_prob, image_size=image_size)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        if self.fce:
            self.lstm = BidirectionalLSTM(
                layer_size=[32], batch_size=self.batch_size, vector_dim=self.g.outSize)

    def forward(self, support_set_images, train_targets, target_image, test_targets, num_ways):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:self.dn
        :return:
        """

        support_images_encoded = self.g(support_set_images.view(-1, *support_set_images.shape[2:]))
        train_embeddings = support_images_encoded.view(*support_set_images.shape[:2], -1)
        target_images_encoded = self.g(target_image.view(-1, *target_image.shape[2:]))
        test_embeddings = target_images_encoded.view(*target_image.shape[:2], -1)
        if self.fce:
            train_embeddings = self.lstm(
                train_embeddings.view(-1, *support_set_images.shape[2:])).view(*support_set_images.shape[:2], -1)
            test_embeddings = self.lstm(
                target_images_encoded.view(-1, *target_image.shape[2:])).view(*target_image.shape[:2], -1)
        loss = matching_loss(train_embeddings,
                             train_targets,
                             test_embeddings,
                             test_targets,
                             num_ways)
        with torch.no_grad():
            # calculate the accuracy
            log_probas = matching_log_probas(train_embeddings,
                                             train_targets,
                                             test_embeddings,
                                             num_ways)
            test_predictions = torch.argmax(log_probas, dim=1)
            accuracy = (test_predictions == test_targets).float()

        return accuracy, loss, train_embeddings
