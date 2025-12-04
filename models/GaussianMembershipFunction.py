import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class LearnableSigmoidMembership(nn.Module):
    def __init__(self, num_sigmoids, feature_dim):
        super(LearnableSigmoidMembership, self).__init__()
        # 隶属函数的可学习参数：中心和斜率
        self.centers = nn.Parameter(torch.randn(num_sigmoids, feature_dim))  # 中心
        self.slopes = nn.Parameter(torch.abs(torch.randn(num_sigmoids, feature_dim)))  # 斜率，确保为正

    def forward(self, x):
        # 计算 sigmoid 隶属度
        # memberships = torch.sigmoid(self.slopes.unsqueeze(0) * (x - self.centers.unsqueeze(0)))
        # return memberships

        x_expanded = x.unsqueeze(1)  # (B, 1, feature_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_sigmoids, feature_dim)
        slopes_expanded = self.slopes.unsqueeze(0)  # (1, num_sigmoids, feature_dim)

        memberships = torch.sigmoid(slopes_expanded * (x_expanded - centers_expanded))

        # 合并输出形状
        return memberships.view(memberships.size(0), -1)  # (B, num_sigmoids * feature_dim)


class LearnableTrapezoidalMembership(nn.Module):
    def __init__(self, num_trapezoids, feature_dim):
        super(LearnableTrapezoidalMembership, self).__init__()
        # 隶属函数的可学习参数：左边界、右边界和高点
        self.a = nn.Parameter(torch.randn(num_trapezoids, feature_dim))  # 左边界
        self.b = nn.Parameter(torch.randn(num_trapezoids, feature_dim))  # 右边界
        self.c = nn.Parameter(torch.randn(num_trapezoids, feature_dim))  # 高点
        self.d = nn.Parameter(torch.randn(num_trapezoids, feature_dim))  # 高点

    def forward(self, x):
        # memberships = []
        # for a, b, c, d in zip(self.a, self.b, self.c, self.d):
        #     # 计算梯形隶属度
        #     membership = torch.zeros_like(x)
        #     # 计算每个区间的隶属度
        #     membership += torch.clamp((x - a) / (b - a), 0, 1) * (x <= b).float()
        #     membership += torch.clamp((d - x) / (d - c), 0, 1) * (x > c).float() * (x <= d).float()
        #     memberships.append(membership)
        # return torch.stack(memberships, dim=1)

        B = x.size(0)  # 批量大小

        # 扩展维度以便广播
        x_expanded = x.unsqueeze(1)  # (B, 1, feature_dim)
        a_expanded = self.a.unsqueeze(0)  # (1, num_trapezoids, feature_dim)
        b_expanded = self.b.unsqueeze(0)  # (1, num_trapezoids, feature_dim)
        c_expanded = self.c.unsqueeze(0)  # (1, num_trapezoids, feature_dim)
        d_expanded = self.d.unsqueeze(0)  # (1, num_trapezoids, feature_dim)

        # 计算每个梯形的隶属度
        memberships = torch.zeros(B, self.a.size(0), x.size(1), device=x.device)

        # 计算每个区间的隶属度
        memberships += torch.clamp((x_expanded - a_expanded) / (b_expanded - a_expanded), 0, 1) * (
                x_expanded <= b_expanded).float()
        memberships += torch.clamp((d_expanded - x_expanded) / (d_expanded - c_expanded), 0, 1) * (
                x_expanded > c_expanded).float() * (x_expanded <= d_expanded).float()

        return memberships.view(memberships.size(0), -1)  # 输出形状 (B, num_trapezoids, feature_dim)


class LearnableGaussianMembershipFunction(nn.Module):
    def __init__(self, num_gaussians, feature_dim):
        super(LearnableGaussianMembershipFunction, self).__init__()
        # 初始化高斯参数为可学习的
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(num_gaussians, feature_dim))  # 随机初始化中心
        self.std_devs = nn.Parameter(torch.abs(torch.randn(num_gaussians, feature_dim)))  # 随机初始化标准差，并取绝对值确保为正

    def forward(self, x):
        # # 计算高斯隶属度
        # memberships = None
        # for c, s in zip(self.centers, self.std_devs):
        #     membership = torch.exp(-((x - c) ** 2) / (2 * (s ** 2)))
        # #     memberships.append(membership)
        # # return torch.stack(memberships, dim=1)
        #     memberships = torch.cat((memberships,membership), dim=1)
        # return  memberships

        # 扩展维度以便广播
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, feature_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_gaussians, feature_dim)
        std_devs_expanded = self.std_devs.unsqueeze(0)  # (1, num_gaussians, feature_dim)

        # 计算高斯隶属度
        memberships = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * (std_devs_expanded ** 2)))
        B = memberships.size()[0]
        return memberships.view(B, self.num_gaussians * self.feature_dim)

        # return memberships.squeeze(1)  # 输出形状为 (batch_size, num_gaussians)


class GaussianMembershipFunction(nn.Module):
    def __init__(self, centers, std_devs):
        super(GaussianMembershipFunction, self).__init__()
        self.centers = centers
        self.std_devs = std_devs

    def forward(self, x):
        # 计算高斯隶属度
        memberships = []
        for c, s in zip(self.centers, self.std_devs):
            membership = torch.exp(-((x - c) ** 2) / (2 * (s ** 2)))
            memberships.append(membership)
        return torch.stack(memberships, dim=1)


class LearnableMembershipFunction(nn.Module):
    def __init__(self, num_feature, feature_dim1, feature_dim):
        super(LearnableMembershipFunction, self).__init__()
        # 高斯隶属函数的中心和标准差
        self.fc1 = nn.Linear(feature_dim1, feature_dim)
        self.gmf = LearnableGaussianMembershipFunction(num_feature, feature_dim)
        self.smf = LearnableSigmoidMembership(num_feature, feature_dim)
        self.tmf = LearnableTrapezoidalMembership(num_feature, feature_dim)

    def forward(self, x):
        x = self.fc1(x)
        Gmemberships = self.gmf(x)
        Smemberships = self.smf(x)
        Tmemberships = self.tmf(x)
        return torch.cat((Gmemberships, Smemberships, Tmemberships), dim=1)


# 模糊神经网络
class FuzzyNeuralNetwork(nn.Module):
    def __init__(self, input_dim=768, membership_num=2,  feature_dim=128):
        super(FuzzyNeuralNetwork, self).__init__()
        self.fuzzy = LearnableMembershipFunction(membership_num, input_dim, feature_dim)
        self.fc1 = nn.Linear(input_dim * 3 * 2, input_dim * 3)
        self.fc2 = nn.Linear(input_dim * 3, input_dim)

    def forward(self, x):
        x = self.fuzzy(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x  # 输出分类概率
