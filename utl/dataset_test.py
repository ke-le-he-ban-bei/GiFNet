import torch
import torchvision.transforms as T
import torchvision.transforms as transforms
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from PIL import Image


class KinshipDataset(Dataset):
    def __init__(self, img_path=None, transform=None, data_name=None, txt=None, fold=None, type='fd'):
        self.root = img_path
        self.transform = transform
        self.type = type

        self.pair_list = []

        self.transform = T.Compose([
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

    def __len__(self):
        return len(self.pair_list)

    def img_transformer(self, image):
        image = Image.fromarray(image)
        img_c_p = self.transform(image)
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
