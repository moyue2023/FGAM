# -*- coding: utf-8 -*-
# 2021.10.25
# 自定义数据集
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, glob, random, csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image, ImageFile
from Binary2img import getBinaryData
import lief
from pe_operations import shift_section_by, padding_PE

# ImageFile.LOAD_TRUNCATED_IMAGES = True


# load_csv('./data/train','train.csv')
# load_csv('./data/validation','val.csv')
# load_csv('./data/test','test.csv')


class MyDataset(Dataset):
    def __init__(self, root, resize, mode, flag, rate):
        """
        """
        super(Dataset, self).__init__()

        self.root = root
        self.resize = resize
        self.name = str(mode) + ".csv"
        self.flag = flag
        self.rate = rate
        self.images = []
        self.labels = []
        # 文件存在，打开获取每一行的内容
        with open(os.path.join(self.root, self.name)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                self.images.append(img)
                self.labels.append(label)
        assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img:\root\name\*.png
        img, label = self.images[idx], self.labels[idx]
        # binary_list: lief.PE.Binary=lief.PE.parse(img)
        # lief_adv: lief.PE.Binary = lief.PE.parse(raw=x_adv)
        filename = img.split("/")[-1]
        img = getBinaryData(img)
        img = bytearray(img)

        if self.flag == "padding":
            embedding_value = 0
            overall_size = int(len(img) * self.rate)
            x, index_to_perturb = padding_PE(img, overall_size)
            img = list(x)
            for i in range(0, len(index_to_perturb)):
                img[index_to_perturb[i]] = random.randint(0, 255)

        if self.flag == "section":
            overall_size = int(len(img) * self.rate)
            x, index_to_perturb = shift_section_by(img, overall_size, 0)
            img = list(x)
            for i in range(0, len(index_to_perturb)):
                img[index_to_perturb[i]] = random.randint(0, 255)

        label = torch.tensor(label)
        return img, label, index_to_perturb, filename


"""

"""

