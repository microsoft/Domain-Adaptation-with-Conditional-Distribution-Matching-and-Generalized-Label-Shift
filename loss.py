# Copyright(c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None, weights=None, device='cuda'):
    softmax_output = input_list[1].detach()
    batch_size = softmax_output.size(0) // 2
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(
            op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        if weights is not None:
            weights = torch.cat((weights.squeeze(), torch.ones((batch_size)).to(device)))
            weight = (source_weight / torch.sum(source_weight).detach().item() +
                    target_weight / torch.sum(target_weight).detach().item()) * weights
        else:
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        if weights is not None:
            weighted_nll_source = - weights * torch.log(ad_out[:batch_size])
            nll_target = - torch.log(1 - ad_out[batch_size:])
            return (torch.mean(weighted_nll_source) + torch.mean(nll_target)) / 2

        return nn.BCELoss()(ad_out, dc_target)


def DANN(features, ad_net, device):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    return nn.BCELoss()(ad_out, dc_target)


def IWDAN(features, ad_net, weights):

    # First batch_size elements of features correspond to source
    # Last  batch_size elements of features correspond to target
    # Each element of ad_out represents the proba of the corresponding feature to be from the source domain
    # For importance sampling, it needs to be put to the log and multiplied by the weight of the corresponding class

    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2

    weighted_nll_source = - weights * torch.log(ad_out[:batch_size])
    nll_target = - torch.log(1 - ad_out[batch_size:])

    return (torch.mean(weighted_nll_source) + torch.mean(nll_target)) / 2


def WDANN(features, ad_net, device, weights=None):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    if weights is None:
        weighted_source = ad_out[:batch_size]
    else:
        weighted_source = ad_out[:batch_size] * weights
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[-1]] * batch_size)).float().to(device)

    # Gradient penalty
    alpha = torch.rand([batch_size, 1]).to(device)
    interpolates = (1 - alpha) * features[batch_size:] + alpha * features[:batch_size]
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = ad_net(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Lambda is 10 in the original WGAN-GP paper
    # return - torch.mean(dc_target * ad_out), ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return - torch.mean(weighted_source - ad_out[batch_size:]) / 2, ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2


def DAN_Linear(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    # Linear version
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68], weights=None):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = gaussian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    loss1 = 0
    if weights is None:
        mult = 1
    for s1 in range(batch_size):
        for s2 in range(s1 + 1, batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            if weights is not None:
                mult = weights[s1] * weights[s2]
            loss1 += mult * joint_kernels[s1, s2] + joint_kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    if weights is None:
        mult1, mult2 = 1, 1
    for s1 in range(batch_size):
        if weights is not None:
            mult1 = weights[s1]
        for s2 in range(batch_size):
            t1, t2 = s1 + batch_size, s2 + batch_size
            if weights is not None:
                mult2 = weights[s2]
            loss2 -= mult1 * joint_kernels[s1, t2] + mult2 * joint_kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2


def JAN_Linear(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = gaussian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    # Linear version
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)


loss_dict = {"DAN": DAN, "DAN_Linear": DAN_Linear, "JAN": JAN,
             "JAN_Linear": JAN_Linear, "IWJAN": JAN, "IWJANORACLE": JAN}
