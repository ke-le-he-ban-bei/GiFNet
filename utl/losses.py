import torch
import numpy as np


# from Track1.utils import *


# def contrastive_loss(x1, x2, beta=0.08):
#     x1_ = torch.cat([x1, x2], dim=1)
#     x2_ = torch.cat([x2, x1], dim=1)
#     x1x2 = torch.cat([x1_, x2_], dim=0)
#     x2x1 = torch.cat([x2_, x1_], dim=0)
#
#     cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1),
#                                          torch.unsqueeze(x1x2, dim=0), dim=2) / beta
#     mask = (1.0 - torch.eye(2 * x1_.size(0))).to('cuda:0')
#     numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
#     denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
#     return -torch.mean(torch.log(numerators / denominators), dim=0)

def contrastive_loss(x1, x2, beta=0.08):
    x1x2 = torch.cat([x1, x2], dim=0)
    x2x1 = torch.cat([x2, x1], dim=0)

    cosine_mat = torch.cosine_similarity(torch.unsqueeze(x1x2, dim=1),
                                         torch.unsqueeze(x1x2, dim=0), dim=2) / beta
    mask = (1.0 - torch.eye(2 * x1.size(0))).to('cuda:0')
    numerators = torch.exp(torch.cosine_similarity(x1x2, x2x1, dim=1) / beta)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)
    return -torch.mean(torch.log(numerators / denominators), dim=0)


def contrastive_loss_(x):
    index = int(x.size(0) / 2)
    x1 = x[:index, :]
    x2 = x[index:, :]
    return contrastive_loss(x1, x2)


def fuse_map_contrastive_loss_(xp, xn, beta=0.08, device='cuda:0'):
    xpn = torch.cat([xp, xn], dim=0)

    cosine_mat = torch.cosine_similarity(torch.unsqueeze(xpn, dim=1),
                                         torch.unsqueeze(xpn, dim=0), dim=2) / beta
    mask = (1.0 - torch.eye(2 * xp.size(0))).to(device)
    denominators = torch.sum(torch.exp(cosine_mat) * mask, dim=1)

    mask_ = (1.0 - torch.eye(xp.size(0))).to(device)
    cosine_mat_pp = torch.cosine_similarity(torch.unsqueeze(xp, dim=1),
                                            torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_pp = torch.sum(torch.exp(cosine_mat_pp) * mask_, dim=1)

    cosine_mat_nn = torch.cosine_similarity(torch.unsqueeze(xn, dim=1),
                                            torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_nn = torch.sum(torch.exp(cosine_mat_nn) * mask_, dim=1)

    return -torch.mean(torch.log(torch.cat((denominators_pp, denominators_nn), dim=0) / denominators), dim=0)


def fuse_map_contrastive_loss_2(xp, xn, beta=0.08):
    mask_ = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    cosine_mat_pn = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_pn = torch.sum(torch.exp(cosine_mat_pn), dim=1)

    cosine_mat_np = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_np = torch.sum(torch.exp(cosine_mat_np), dim=1)

    cosine_mat_pp = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_pp = torch.sum(torch.exp(cosine_mat_pp) * mask_, dim=1)

    cosine_mat_nn = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_nn = torch.sum(torch.exp(cosine_mat_nn) * mask_, dim=1)

    return torch.mean(torch.cat((denominators_pn, denominators_np), dim=0) /
                      torch.cat((denominators_pp, denominators_nn), dim=0) / 10, dim=0)


def fuse_map_contrastive_loss_3(xp, xn, beta=0.08):
    mask_ = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    cosine_mat_pn = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_pn = torch.sum(torch.exp(cosine_mat_pn), dim=1)

    cosine_mat_np = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_np = torch.sum(torch.exp(cosine_mat_np), dim=1)

    cosine_mat_pp = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_pp = torch.sum(torch.exp(cosine_mat_pp) * mask_, dim=1)

    cosine_mat_nn = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_nn = torch.sum(torch.exp(cosine_mat_nn) * mask_, dim=1)
    loss = torch.cat((denominators_pp, denominators_nn), dim=0) / torch.cat((denominators_pn, denominators_np), dim=0)

    return -torch.mean(torch.log(loss), dim=0), loss


def fuse_map_contrastive_loss_4(xp, xn, beta=0.08):
    mask_ = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    cosine_mat_pn = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_pn = torch.sum(torch.exp(cosine_mat_pn), dim=1)

    cosine_mat_np = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_np = torch.sum(torch.exp(cosine_mat_np), dim=1)

    cosine_mat_pp = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_pp = torch.sum(torch.exp(cosine_mat_pp) * mask_, dim=1)

    cosine_mat_nn = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_nn = torch.sum(torch.exp(cosine_mat_nn) * mask_, dim=1)
    loss = torch.cat((denominators_pn, denominators_np), dim=0) / torch.cat((denominators_pp, denominators_nn), dim=0)
    return torch.mean(torch.atan(loss), dim=0), loss


def fuse_map_contrastive_loss(x, device):
    # xp = [i for i in range(len(label)) if label[i] == 1]
    # xn = [i for i in range(len(label)) if label[i] == 0]
    # # 指定要获取的第一维索引
    # indices_p = torch.tensor(xp)
    # xp_emb = torch.index_select(x, 0, indices_p)
    # indices_n = torch.tensor(xn)
    # xn_emb = torch.index_select(x, 0, indices_n)
    index = int(x.size(0) / 2)
    x_p = x[:index, :]
    x_n = x[index:, :]
    loss = fuse_map_contrastive_loss_(x_p, x_n, device=device)
    return loss


def fuse_map_contrastive_loss2(x):
    index = int(x.size(0) / 2)
    x_p = x[:index, :]
    x_n = x[index:, :]
    loss = fuse_map_contrastive_loss_2(x_p, x_n)
    return loss


def fuse_map_contrastive_loss3(x):
    index = int(x.size(0) / 2)
    x_p = x[:index, :]
    x_n = x[index:, :]
    loss = fuse_map_contrastive_loss_3(x_p, x_n)
    return loss


def fuse_map_contrastive_loss4(x):
    index = int(x.size(0) / 2)
    x_p = x[:index, :]
    x_n = x[index:, :]
    loss = fuse_map_contrastive_loss_4(x_p, x_n)
    return loss


def fuse_map_contrastive_loss_5(xp, xn, beta=0.08):
    mask_ = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    cosine_mat_pn = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_pn = torch.sum(torch.exp(cosine_mat_pn), dim=1)

    cosine_mat_np = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_np = torch.sum(torch.exp(cosine_mat_np), dim=1)

    cosine_mat_pp = torch.cosine_similarity(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xp, dim=0), dim=2) / beta
    denominators_pp = torch.mean(torch.exp(cosine_mat_pp) * mask_, dim=1)  # torch.mean

    cosine_mat_nn = torch.cosine_similarity(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xn, dim=0), dim=2) / beta
    denominators_nn = torch.mean(torch.exp(cosine_mat_nn) * mask_, dim=1)  # torch.mean
    loss = torch.cat((denominators_pp, denominators_nn), dim=0) / torch.cat((denominators_pn, denominators_np), dim=0)

    return -torch.mean(torch.log(loss), dim=0)


def fuse_map_contrastive_loss5(x):
    index = int(x.size(0) / 2)
    x_p = x[:index, :]
    x_n = x[index:, :]
    loss = fuse_map_contrastive_loss_5(x_p, x_n)
    return loss


def pre_contrastive_loss(xp, xn, beta=0.08):
    mask_ = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    cosine_mat_pn = abs(torch.unsqueeze(xp, dim=1)-torch.unsqueeze(xn, dim=0)) / beta
    denominators_pn = torch.sum(torch.exp(cosine_mat_pn), dim=1)

    cosine_mat_np = abs(torch.unsqueeze(xn, dim=1)-torch.unsqueeze(xp, dim=0)) / beta
    denominators_np = torch.sum(torch.exp(cosine_mat_np), dim=1)

    cosine_mat_pp = abs(torch.unsqueeze(xp, dim=1) - torch.unsqueeze(xp, dim=0)) / beta
    denominators_pp = torch.sum(torch.exp(cosine_mat_pp) * mask_, dim=1)  # torch.mean

    cosine_mat_nn = abs(torch.unsqueeze(xn, dim=1)-torch.unsqueeze(xn, dim=0)) / beta
    denominators_nn = torch.sum(torch.exp(cosine_mat_nn) * mask_, dim=1)  # torch.mean
    loss = torch.cat((denominators_pp, denominators_nn), dim=0) / torch.cat((denominators_pn, denominators_np), dim=0)

    return torch.mean(loss, dim=0)/0.0001


def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=2) + 1e-8)


def custom_loss_function1(x, beta=0.08):
    index = int(x.size(0) / 2)
    xp = x[:index, :]
    xn = x[index:, :]
    xpn = torch.cat([xp, xn], dim=0)

    euclidean_mat = euclidean_distance(torch.unsqueeze(xpn, dim=1), torch.unsqueeze(xpn, dim=0)) / beta
    mask1 = (1.0 - torch.eye(2 * xp.size(0))).to('cuda:0')
    denominators = torch.mean(euclidean_mat*mask1, dim=1)

    mask2 = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    euclidean_mat_pp = euclidean_distance(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xp, dim=0)) / beta
    denominators_pp = torch.mean(euclidean_mat_pp*mask2, dim=1)

    euclidean_mat_nn = euclidean_distance(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xn, dim=0)) / beta
    denominators_nn = torch.mean(euclidean_mat_nn*mask2, dim=1)

    return torch.mean(torch.cat((denominators_pp, denominators_nn), dim=0) / denominators, dim=0)


def custom_loss_function2(x, beta=0.08):
    index = int(x.size(0) / 2)
    xp = x[:index, :]
    xn = x[index:, :]

    mask_ = (1.0 - torch.eye(xp.size(0))).to('cuda:0')

    cosine_mat_pn = euclidean_distance(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xn, dim=0)) / beta
    denominators_pn = torch.sum(cosine_mat_pn, dim=1)

    cosine_mat_np = euclidean_distance(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xp, dim=0)) / beta
    denominators_np = torch.sum(cosine_mat_np, dim=1)

    cosine_mat_pp = euclidean_distance(torch.unsqueeze(xp, dim=1), torch.unsqueeze(xp, dim=0)) / beta
    denominators_pp = torch.sum(cosine_mat_pp * mask_, dim=1)

    cosine_mat_nn = euclidean_distance(torch.unsqueeze(xn, dim=1), torch.unsqueeze(xn, dim=0)) / beta
    denominators_nn = torch.sum(cosine_mat_nn * mask_, dim=1)

    return torch.mean(torch.cat((denominators_pn, denominators_np), dim=0) /
                      torch.cat((denominators_pp, denominators_nn), dim=0) / 10, dim=0)
