import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from .dataset import ImgDataset

def read_spilt_data(data_path):
    assert os.path.exists(data_path), "data path:{} does not exist".format(data_path)

    font_class = glob(os.path.join(data_path, '*'))
    num_classes = len(font_class)
    font_class.sort()
    font_class_indices = dict((k, v) for v, k in enumerate(font_class))
    # print(font_class_indices)

    

    # 80% fonts for train, 20% font for eval

    train_font = []
    val_font = []

    spilt_point = random.sample(font_class, k=int(len(font_class) * 0.8))
    for font_path in font_class:
        if font_path in spilt_point:
            train_font.append(font_path)
        else:
            val_font.append(font_path)

    return train_font, val_font, num_classes

    # each font, 80% for train, 20% for eval

    # train_data = []
    # train_label = []
    # val_data = []
    # val_label = []

    # for cla in font_class:
    #     img = glob(os.path.join(cla, '*'))
    #     # print(img)
    #     img_class = font_class_indices[cla]
    #     # print(img_class)

    #     spilt_point = random.sample(img, k=int(len(img) * 0.8))
    #     print(len(spilt_point))
        
    #     for img_path in img:
    #         if img_path in spilt_point:
    #             train_data.append(img_path)
    #             train_label.append(img_class)
    #         else:
    #             val_data.append(img_path)
    #             val_label.append(img_class)

    # return train_data, train_label, val_data, val_label, num_classes

def get_loader(use_ddp, batch_size, num_workers, cfgs):
    train_font, val_font, num_classes = read_spilt_data(cfgs.data_path)

    # font_path, num_classes, stroke_path, four_corner_path, unicode_path
    train_dataset = ImgDataset(train_font, num_classes, cfgs.stroke_path, cfgs.four_corner_path, cfgs.unicode_path)
    val_dataset = ImgDataset(val_font, num_classes, cfgs.stroke_path, cfgs.four_corner_path, cfgs.unicode_path)

    if use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            sampler=train_sampler
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=num_workers,
            sampler=val_sampler
        )
    