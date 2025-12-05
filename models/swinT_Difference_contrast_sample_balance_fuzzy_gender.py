import torch
import torch.nn as nn
import torch.nn.functional as F
from SwinT.swin_transformer import SwinTransformer
from models.GaussianMembershipFunction import LearnableMembershipFunction
from timm.models.layers import trunc_normal_
import random


def create_negative_samples(n):
    # 生成负样本下标索引
    negative_samples = []

    for i in range(n):
        while True:
            # 随机选择一个新的下标
            rand_index = random.randint(0, n - 1)
            # 如果随机下标不是当前下标，则构造负样本
            if rand_index != i:
                negative_samples.append(rand_index)
                break
    return negative_samples


class ProbInferNet(nn.Module):
    def __init__(self, input_dim=768, num_classes=2):  # num_classes 表示分类数
        super(ProbInferNet, self).__init__()
        self.fc1 = nn.Linear(input_dim * 3 * 2, input_dim * 3)
        self.fc2 = nn.Linear(input_dim * 3, input_dim)
        self.fc3 = nn.Linear(input_dim, num_classes)  # 128 维特征，每个特征有3个模糊化值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  # 输出分类概率


class GenderInformation(nn.Module):
    def __init__(self, input_dim=768, num_f=2):  # num_classes 表示分类数
        super(GenderInformation, self).__init__()
        self.diff_inf = LearnableMembershipFunction(num_f, 768, 768)
        self.share_inf = LearnableMembershipFunction(num_f, 768, 768)
        self.diff_fc = nn.Linear(input_dim * 3 * 2, input_dim)
        self.share_fc = nn.Linear(input_dim * 3 * 2, input_dim)

    def forward(self, x):
        diff = F.relu(self.diff_fc(self.diff_inf(x)))
        share = F.relu(self.share_fc(self.share_inf(x)))

        return diff, share


class KinClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(KinClassifier, self).__init__()
        # self.cnn = CNNFeatureExtractor()
        self.feature_extractor = SwinTransformer(
            img_size=64,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=4,
            patch_size=4,
            num_classes=0,
            in_chans=9)

        self.fuzzy = LearnableMembershipFunction(2, 768 * 3, 768)
        self.gender_inf = GenderInformation()

        self.infer_net_gender = nn.Sequential(
            nn.Linear(in_features=768, out_features=128),  # 第一个全连接层
            nn.ReLU(),  # 激活函数
            nn.Dropout(p=0.5),  # Dropout层（可选，防止过拟合）
            nn.Linear(in_features=128, out_features=2)  # 第二个全连接层
        )

        self.infer_net = ProbInferNet(num_classes=num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2, train=True):
        # 提取特征

        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        x1_diff_inf, x1_share_ifn = self.gender_inf(x1)
        x2_diff_inf, x2_share_ifn = self.gender_inf(x2)
        x1_kin = x1 * x1_diff_inf
        x1_kin_ = x1_kin + x1_share_ifn

        x2_kin = x2 * x2_diff_inf
        x2_kin_ = x2_kin + x2_share_ifn

        x1_gender = x1 - x1_kin + x1_share_ifn
        x2_gender = x2 - x2_kin + x2_share_ifn

        if train:
            negative_sample_index = create_negative_samples(x1.size(0))
            x2_n = x2_kin_[negative_sample_index]

            x1_kin__ = torch.cat((x1_kin_, x1_kin_), dim=0)
            x2_kin__ = torch.cat((x2_kin_, x2_n), dim=0)
            x = torch.cat((torch.abs(x1_kin__ - x2_kin__), torch.mul((x1_kin__ - x2_kin__), (x1_kin__ - x2_kin__)),
                           torch.mul(x1_kin__, x2_kin__)), dim=1)

            # # 通过模糊神经网络进行分类
            # x = self.fuzzy(x)
            # output = self.infer_net(x)
            #
            #
            #
            # gender = torch.cat((x1_gender, x2_gender), dim=0)
            # gender_output = self.infer_net_gender(gender)
            #
            # return x1_kin_, x2_kin_, output, gender_output
        else:
            x = torch.cat((torch.abs(x1_kin_ - x2_kin_), torch.mul((x1_kin_ - x2_kin_), (x1_kin_ - x2_kin_)),
                           torch.mul(x1_kin_, x2_kin_)), dim=1)

        # 通过模糊神经网络进行分类
        x = self.fuzzy(x)
        output = self.infer_net(x)

        # x1_gender = 1 - x1_kin + x1_share_ifn
        # x2_gender = 1 - x2_kin + x2_share_ifn

        gender = torch.cat((x1_gender, x2_gender), dim=0)
        gender_output = self.infer_net_gender(gender)

        return x1_kin_, x2_kin_, output, gender_output
