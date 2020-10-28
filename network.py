# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn.functional as F
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


logger = logging.getLogger(__name__)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = './alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model


# convnet without the last layer
class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(AlexNetFc, self).__init__()
    model_alexnet = alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(4096, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_alexnet.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params": self.features.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2},
                              {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.classifier.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
    else:
        parameter_list = [
            {"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
    return parameter_list


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34,
               "ResNet50": models.resnet50, "ResNet101": models.resnet101, "ResNet152": models.resnet152}


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000, ma=0.0):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

        self.im_weights_update = create_im_weights_update(self, ma, class_num)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2},
                                  {"params": self.bottleneck.parameters(
                                  ), "lr_mult": 10, 'decay_mult': 2},
                                  {"params": self.fc.parameters(), "lr_mult": 10,
                                   'decay_mult': 2},
                                  {"params": self.im_weights, "lr_mult": 10, 'decay_mult': 2}]
            else:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2},
                                  {"params": self.fc.parameters(), "lr_mult": 10,
                                   'decay_mult': 2},
                                  {"params": self.im_weights, "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.im_weights, "lr_mult": 10, 'decay_mult': 2}]
        return parameter_list


vgg_dict = {"VGG11": models.vgg11, "VGG13": models.vgg13, "VGG16": models.vgg16, "VGG19": models.vgg19,
            "VGG11BN": models.vgg11_bn, "VGG13BN": models.vgg13_bn, "VGG16BN": models.vgg16_bn, "VGG19BN": models.vgg19_bn}


class VGGFc(nn.Module):
  def __init__(self, vgg_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module(
            "classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(4096, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(4096, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_vgg.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params": self.features.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.classifier.parameters(
                              ), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.bottleneck.parameters(
                              ), "lr_mult": 10, 'decay_mult': 2},
                              {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        else:
            parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.classifier.parameters(
                              ), "lr_mult": 1, 'decay_mult': 2},
                              {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}]
    else:
        parameter_list = [
            {"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
    return parameter_list


class LeNet(nn.Module):
    def __init__(self, ma=0.0):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )

        class_num = 10

        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, class_num)
        self.__in_features = 500

        self.im_weights_update = create_im_weights_update(self, ma, class_num)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, sigmoid=True):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = sigmoid
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    if self.sigmoid:
        y = nn.Sigmoid()(y)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, inputs):
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(tensor):
    return GradientReversalLayer.apply(tensor)


class ResNet50Fc(nn.Module):
  def __init__(self, ma=0.0, class_num=31, **kwargs):
    super(ResNet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained=True)
    self.conv1 = model_resnet50.conv1
    self.bn1 = model_resnet50.bn1
    self.relu = model_resnet50.relu
    self.maxpool = model_resnet50.maxpool
    self.layer1 = model_resnet50.layer1
    self.layer2 = model_resnet50.layer2
    self.layer3 = model_resnet50.layer3
    self.layer4 = model_resnet50.layer4
    self.avgpool = model_resnet50.avgpool
    self.__in_features = model_resnet50.fc.in_features

    self.im_weights_update = create_im_weights_update(self, ma, class_num)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features


class LeNetMMD(nn.Module):
    def __init__(self, ma=0.0, **kwargs):
        super(LeNetMMD, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        class_num = 10

        self.fc_params = nn.Sequential(
            nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        # self.__in_features = 500
        self.__in_features = 800

        self.im_weights_update = create_im_weights_update(self, ma, class_num)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        # x = self.fc_params(x)
        return x

    def output_num(self):
        return self.__in_features


network_dict = {"ResNet50" : ResNet50Fc, "LeNet": LeNetMMD}


def create_im_weights_update(class_inst, ma, class_num):

    # Label importance weights.
    class_inst.ma = ma
    class_inst.im_weights = nn.Parameter(
        torch.ones(class_num, 1), requires_grad=False)

    def im_weights_update(source_y, target_y, cov, device, inst=class_inst):
        """
        Solve a Quadratic Program to compute the optimal importance weight under the generalized label shift assumption.
        :param source_y:    The marginal label distribution of the source domain.
        :param target_y:    The marginal pseudo-label distribution of the target domain from the current classifier.
        :param cov:         The covariance matrix of predicted-label and true label of the source domain.
        :param device:      Device of the operation.
        :return:
        """
        # Convert all the vectors to column vectors.
        dim = cov.shape[0]
        source_y = source_y.reshape(-1, 1).astype(np.double)
        target_y = target_y.reshape(-1, 1).astype(np.double)
        cov = cov.astype(np.double)

        P = matrix(np.dot(cov.T, cov), tc="d")
        q = -matrix(np.dot(cov, target_y), tc="d")
        G = matrix(-np.eye(dim), tc="d")
        h = matrix(np.zeros(dim), tc="d")
        A = matrix(source_y.reshape(1, -1), tc="d")
        b = matrix([1.0], tc="d")
        sol = solvers.qp(P, q, G, h, A, b)
        new_im_weights = np.array(sol["x"])

        # EMA for the weights
        inst.im_weights.data = (1 - inst.ma) * torch.tensor(
            new_im_weights, dtype=torch.float32).to(device) + inst.ma * inst.im_weights.data

    return im_weights_update
