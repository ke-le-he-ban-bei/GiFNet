import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import cv2
from scipy.io import loadmat
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from skimage.feature import local_binary_pattern
import numpy as np
from PIL import Image


class KinshipDataset(Dataset):
    def __init__(self, img_path=None, transform=None, data_name=None, txt=None, fold=None, color=None, type='fd'):
        self.root = img_path
        self.transform = transform
        self.type = type

        if color == 'rgb':
            self.img_transformer = self.img_transformer_rgb
        elif color == 'hsv':
            self.img_transformer = self.img_transformer_hsv
        elif color == 'lab':
            self.img_transformer = self.img_transformer_lab
        elif color == 'rgb_hsv':
            self.img_transformer = self.img_transformer_rgb_hsv
        elif color == 'rgb_lab':
            self.img_transformer = self.img_transformer_rgb_lab
        elif color == 'hsv_lab':
            self.img_transformer = self.img_transformer_hsv_lab
        elif color == 'rgb_hsv_lab':
            self.img_transformer = self.img_transformer_rgb_hsv_lab
        else:
            raise ValueError("Unsupported transform_type:", color)
        self.pair_list = []

        self.transform_img_rgb = T.Compose([
            T.Lambda(self.convert_to_rgb),
            T.ToTensor(),
        ])

        self.transform_img_hsv = T.Compose([
            T.Lambda(self.convert_to_hsv),
            T.ToTensor(),
        ])

        self.transform_img_ycbcr = T.Compose([
            T.Lambda(self.convert_to_ycbcr),
            T.ToTensor(),
        ])

        for index, name in enumerate(data_name):
            self.txt = txt[index]
            self.data_name = name
            self._prepare_2(fold)

    def _prepare_2(self, fold):
        if self.data_name == 'KFWI' or self.data_name == 'KFWII':
            with open(self.txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                if line[0] in fold:
                    parent_path = line[2].replace('D:/work/data/kinshipdatabase/', 'D:/other/')
                    child_path = line[3].replace('\n', '').replace('D:/work/data/kinshipdatabase/', 'D:/other/')
                    type_name = parent_path.split('/')[4]
                    type_ = None
                    gender_ = None

                    if type_name == 'father-dau':
                        type_ = 'fd'
                        gender_ = [0, 1]
                    elif type_name == 'father-son':
                        type_ = 'fs'
                        gender_ = [0, 0]
                    elif type_name == 'mother-dau':
                        type_ = 'md'
                        gender_ = [1, 1]
                    elif type_name == 'mother-son':
                        type_ = 'ms'
                        gender_ = [1, 0]
                    self.pair_list.append((parent_path, child_path, int(line[1]), type_, gender_))

        elif self.data_name == 'TSKinFace':
            with open(self.txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                if line[0] in fold:
                    parent_path = line[1].replace('D:/work/data/kinshipdatabase/', 'D:/other/')
                    child_path = line[2].replace('D:/work/data/kinshipdatabase/', 'D:/other/')
                    label = int(line[3].split('\n')[0])
                    type_name_p = parent_path[-5]
                    type_name_c = child_path[-5]
                    type_name = type_name_p + type_name_c
                    type_ = None
                    if type_name == 'FD':
                        type_ = 'fd'
                        gender_ = [0, 1]
                    elif type_name == 'FS':
                        type_ = 'fs'
                        gender_ = [0, 0]
                    elif type_name == 'MD':
                        type_ = 'md'
                        gender_ = [1, 1]
                    elif type_name == 'MS':
                        type_ = 'ms'
                        gender_ = [1, 0]

                    self.pair_list.append((parent_path, child_path, label, type_, gender_))

    def convert_to_rgb(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def convert_to_hsv(self, img):
        return img.convert('HSV')

    def rgb_to_xyz(self, rgb):
        # 将RGB转换为XYZ
        rgb = np.asarray(rgb) / 255.0
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        rgb = rgb * 100.0
        matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                           [0.2126729, 0.7151522, 0.0721750],
                           [0.0193339, 0.1191920, 0.9503041]])
        xyz = np.dot(rgb, matrix.T)
        return xyz

    def xyz_to_lab(self, xyz):
        # 将XYZ转换为Lab
        xyz = xyz / np.array([95.047, 100.000, 108.883])
        xyz = np.where(xyz > 0.008856, xyz ** (1 / 3), (xyz * 903.3 + 16) / 116)
        lab = np.array(
            [116 * xyz[:, :, 1] - 16, 500 * (xyz[:, :, 0] - xyz[:, :, 1]), 200 * (xyz[:, :, 1] - xyz[:, :, 2])])

        return lab

    def rgb_to_lab(self, rgb):
        # 将RGB转换为Lab
        xyz = self.rgb_to_xyz(rgb)
        lab = self.xyz_to_lab(xyz)

        # 归一化
        lab_l = lab[0] / 100.
        lab_a = (lab[1] + 128.) / (127. + 128.)
        lab_b = (lab[2] + 128.) / (127. + 128.)
        lab_l = torch.from_numpy(lab_l).float()
        lab_a = torch.from_numpy(lab_a).float()
        lab_b = torch.from_numpy(lab_b).float()
        lab = torch.stack([lab_l, lab_a, lab_b], dim=0)
        return lab

    def rgb(self, rgb):
        image = rgb.astype(np.float32)
        # 计算每个像素的RGB之和
        sum_ = image.sum(axis=2, keepdims=True)  # 在第2个维度上求和，并保持维度为3
        smooth_factor = 1e-10  # 可以根据具体情况调整平滑因子的大小
        sum_ = sum_ + smooth_factor
        # 归一化每个像素的RGB值
        normalized_image = torch.from_numpy(image / sum_).float()
        image = normalized_image.permute(2, 0, 1)
        return image

    def __len__(self):
        return len(self.pair_list)

    def img_transformer_rgb(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform_img_rgb(image)
        return img_c_p

    def img_transformer_hsv(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform_img_hsv(image)
        return img_c_p

    def img_transformer_lab(self, image):
        image = Image.fromarray(image)
        img_c_p = self.rgb_to_lab(image)
        return img_c_p

    def img_transformer_rgb_hsv(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform_img_rgb(image)
        tt_p = self.transform_img_hsv(image)
        img_c_p = torch.cat((img_c_p, tt_p), dim=0)
        return img_c_p

    def img_transformer_rgb_lab(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform_img_rgb(image)
        tt_p = self.rgb_to_lab(image)
        img_c_p = torch.cat((img_c_p, tt_p), dim=0)
        return img_c_p

    def img_transformer_hsv_lab(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform_img_hsv(image)
        tt_p = self.rgb_to_lab(image)
        img_c_p = torch.cat((img_c_p, tt_p), dim=0)
        return img_c_p

    def img_transformer_rgb_hsv_lab(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform_img_rgb(image)
        tt_p = self.transform_img_hsv(image)
        img_c_p = torch.cat((img_c_p, tt_p), dim=0)
        tt_p = self.rgb_to_lab(image)
        img_c_p = torch.cat((img_c_p, tt_p), dim=0)
        return img_c_p

    def __getitem__(self, index):  # 做翻转缩放增强
        parent_path_p, child_path_p, label_p, x_p, gender_ = self.pair_list[index]

        image_p_p = cv2.imread(parent_path_p, cv2.IMREAD_COLOR)

        image_c_p = cv2.imread(child_path_p, cv2.IMREAD_COLOR)

        image_p_p = cv2.resize(image_p_p, (64, 64))
        image_c_p = cv2.resize(image_c_p, (64, 64))

        p_p = self.img_transformer(image_p_p)
        c_p = self.img_transformer(image_c_p)

        return p_p, c_p, label_p, x_p, gender_[0], gender_[1]
