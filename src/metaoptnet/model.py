import torch
import torch.nn as nn
import torch.nn.functional as F
from src.metaoptnet.classification_heads import ClassificationHead
from src.metaoptnet.embedding_networks import ProtoNetEmbedding, R2D2Embedding, resnet12


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy


class MetaOptNet(nn.Module):
    def __init__(self, dataset, network, head, train_way, train_shot, train_query):
        """
        This is our main network
        :param embedding_network:
        """
        super(MetaOptNet, self).__init__()
        self.cls_head = ClassificationHead(base_learner=head).cuda()

        if network == 'ProtoNet':
            self.embedding_network = ProtoNetEmbedding().cuda()
        elif network == 'R2D2':
            self.embedding_network = R2D2Embedding().cuda()
        elif network == 'ResNet':
            if dataset == 'miniImageNet' or dataset == 'tieredImageNet':
                self.embedding_network = resnet12(
                    avg_pool=False, drop_rate=0.1, dropblock_size=5).cuda()
                self.embedding_network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
            else:
                self.embedding_network = resnet12(
                    avg_pool=False, drop_rate=0.1, dropblock_size=2).cuda()
        else:
            print("Cannot recognize the network type")
            assert(False)

        self.episodes_per_batch = 1
        self.train_n_support = train_way * train_shot
        self.train_n_query = train_way * train_query
        self.train_way = train_way
        self.train_shot = train_shot
        self.eps = 0.0

    def forward(self, support_set_images, support_set_y, target_image, target_y, dpp=False):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
        :param target_image: shape[batch_size,num_channels,image_size,image_size]
        :param target_y:self.dn
        :return:
        """
        if support_set_images.size()[1] == 1:
            support_set_images = support_set_images.repeat(1, 3, 1, 1)
            target_image = target_image.repeat(1, 3, 1, 1)
        emb_support = self.embedding_network(support_set_images.reshape(
            [-1] + list(support_set_images.shape[-3:])))
        emb_support = emb_support.reshape(self.episodes_per_batch, self.train_n_support, -1)
        if dpp:
            return emb_support

        emb_query = self.embedding_network(
            target_image.reshape([-1] + list(target_image.shape[-3:])))
        emb_query = emb_query.reshape(self.episodes_per_batch, self.train_n_query, -1)

        logit_query = self.cls_head(emb_query, emb_support, support_set_y,
                                    self.train_way, self.train_shot)

        smoothed_one_hot = one_hot(target_y.reshape(-1), self.train_way)
        smoothed_one_hot = smoothed_one_hot * \
            (1 - self.eps) + (1 - smoothed_one_hot) * self.eps / (self.train_way - 1)

        log_prb = F.log_softmax(logit_query.reshape(-1, self.train_way), dim=1)
        crossentropy_loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        crossentropy_loss = crossentropy_loss.mean()

        accuracy = count_accuracy(logit_query.reshape(-1, self.train_way),
                                  target_y.reshape(-1)).item()
        return accuracy, crossentropy_loss, emb_support
