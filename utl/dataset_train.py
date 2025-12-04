import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
import os
import random
from torch.utils.data import Dataset
import cv2

import numpy as np
from PIL import Image


class KinshipDataset(Dataset):
    def __init__(self, img_path=None, transform=None, data_name=None, txt=None, fold=None):
        self.root = img_path
        self.transform = transform

        self.pair_list = []
        self.transform_img = T.Compose([
            T.Lambda(self.convert_to_rgb),
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
                    if int(line[1]) == 1:
                        if type_name == 'father-dau':
                            type_ = 'fd'
                            self.pair_list.append((parent_path, child_path, int(line[1]), type_, 0, 1))
                        elif type_name == 'father-son':
                            type_ = 'fs'
                            self.pair_list.append((parent_path, child_path, int(line[1]), type_, 0, 0))
                        elif type_name == 'mother-dau':
                            type_ = 'md'
                            self.pair_list.append((parent_path, child_path, int(line[1]), type_, 1, 1))
                        elif type_name == 'mother-son':
                            type_ = 'ms'
                            self.pair_list.append((parent_path, child_path, int(line[1]), type_, 1, 0))

        elif self.data_name == 'TSKinFace':
            with open(self.txt, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                if line[0] in fold:
                    parent_path = line[1].replace('D:/work/data/kinshipdatabase/', 'D:/ZQ/datasets/')
                    child_path = line[2].replace('D:/work/data/kinshipdatabase/', 'D:/ZQ/datasets/')

                    if int(line[3]) == 1:
                        if parent_path[-5] == 'F':
                            gender1 = 0
                            type_ = 'f'
                        else:
                            gender1 = 1
                            type_ = 'm'
                        if child_path[-5] == 'S':
                            gender2 = 0
                            type__ = 's'
                        else:
                            gender2 = 1
                            type__ = 'd'

                        self.pair_list.append(
                            (parent_path, child_path, int(line[3].split('\n')[0]), type_ + type__, gender1, gender2))

    def convert_to_rgb(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def __len__(self):
        return len(self.pair_list)

    def img_transformer(self, image):
        image = image.astype(np.float32)

        # 添加高斯噪声
        if np.random.rand() < 0.5:
            mean, std = 0, 25
            noise = np.random.normal(mean, std, image.shape)
            image += noise
            image = np.clip(image, 0, 255)
        image = Image.fromarray(np.uint8(image))

        # 多通道组合
        # # RGB
        img_c_p = self.transform(image)

        # 随机水平翻转
        horizontal_flip_p = T.RandomHorizontalFlip()
        img_c_p = horizontal_flip_p(img_c_p)

        return img_c_p

    def __getitem__(self, index):  # 做翻转缩放增强
        parent_path_p, child_path_p, label_p, x_p, gender_p, gender_c = self.pair_list[index]

        image_p_p = cv2.imread(parent_path_p, cv2.IMREAD_COLOR)

        image_c_p = cv2.imread(child_path_p, cv2.IMREAD_COLOR)

        # image_p_p = cv2.resize(image_p_p, (224, 224))
        # image_c_p = cv2.resize(image_c_p, (224, 224))

        p_p = self.img_transformer(image_p_p)
        c_p = self.img_transformer(image_c_p)

        return p_p, c_p, label_p, gender_p, gender_c, x_p
