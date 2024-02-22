import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

# Embedding network used in Meta-learning with differentiable closed-form solvers
# (Bertinetto et al., in submission to NIPS 2018).
# They call the ridge rigressor version as "Ridge Regression Differentiable Discriminator (R2D2)."

# Note that they use a peculiar ordering of functions, namely conv-BN-pooling-lrelu,
# as opposed to the conventional one (conv-BN-lrelu-pooling).

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):
        # shape: (bsize, channels, height, width)

        if self.training:
            batch_size, channels, height, width = x.shape

            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height -
                                    (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * \
                block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)

        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size,
                                                                 self.block_size).reshape(-1),  # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size),  # - left_padding
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)

        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask  # [:height, :width]
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) *
                                (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * \
                    feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(
            block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                      downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model


def R2D2_conv_block(in_channels, out_channels, retain_activation=True, keep_prob=1.0):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(2)
    )
    if retain_activation:
        block.add_module("LeakyReLU", nn.LeakyReLU(0.1))

    if keep_prob < 1.0:
        block.add_module("Dropout", nn.Dropout(p=1 - keep_prob, inplace=False))

    return block


class R2D2Embedding(nn.Module):
    def __init__(self, x_dim=3, h1_dim=96, h2_dim=192, h3_dim=384, z_dim=512,
                 retain_last_activation=False):
        super(R2D2Embedding, self).__init__()

        self.block1 = R2D2_conv_block(x_dim, h1_dim)
        self.block2 = R2D2_conv_block(h1_dim, h2_dim)
        self.block3 = R2D2_conv_block(h2_dim, h3_dim, keep_prob=0.9)
        # In the last conv block, we disable activation function to boost the classification accuracy.
        # This trick was proposed by Gidaris et al. (CVPR 2018).
        # With this trick, the accuracy goes up from 50% to 51%.
        # Although the authors of R2D2 did not mention this trick in the paper,
        # we were unable to reproduce the result of Bertinetto et al. without resorting to this trick.
        self.block4 = R2D2_conv_block(
            h3_dim, z_dim, retain_activation=retain_last_activation, keep_prob=0.7)

    def forward(self, x):
        b1 = self.block1(x)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        # Flatten and concatenate the output of the 3rd and 4th conv blocks as proposed in R2D2 paper.
        return torch.cat((b3.view(b3.size(0), -1), b4.view(b4.size(0), -1)), 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, retain_activation=True):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if retain_activation:
            self.block.add_module("ReLU", nn.ReLU(inplace=True))
        self.block.add_module("MaxPool2d", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.block(x)
        return out

# Embedding network used in Matching Networks (Vinyals et al., NIPS 2016), Meta-LSTM (Ravi & Larochelle, ICLR 2017),
# MAML (w/ h_dim=z_dim=32) (Finn et al., ICML 2017), Prototypical Networks (Snell et al. NIPS 2017).


class ProtoNetEmbedding(nn.Module):
    def __init__(self, x_dim=3, h_dim=64, z_dim=64, retain_last_activation=True):
        super(ProtoNetEmbedding, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(x_dim, h_dim),
            ConvBlock(h_dim, h_dim),
            ConvBlock(h_dim, h_dim),
            ConvBlock(h_dim, z_dim, retain_activation=retain_last_activation),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
