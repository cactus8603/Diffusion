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

from .dataset import ImgDataSet

def read_spilt_data(data_path):
    assert os.path.exists(data_path), "data path:{} does not exist".format(data_path)

    font_class = glob(os.path.join(data_path, '*'))
    num_classes = len(font_class)
    font_class.sort()
    font_class_indices = dict((k, v) for v, k in enumerate(font_class))
    # print(font_class_indices)

    train_data = []
    train_label = []
    val_data = []
    val_label = []

    for cla in font_class:
        img = glob(os.path.join(cla, '*'))
        # print(img)
        img_class = font_class_indices[cla]
        # print(img_class)

        spilt_point = random.sample(img, k=int(len(img) * data_path))
        
        for img_path in img:
            if img_path in spilt_point:
                train_data.append(img_path)
                train_label.append(img_class)
            else:
                val_data.append(img_path)
                val_label.append(img_class)

    return train_data, train_label, val_data, val_label, num_classes

def get_loader(data_path, use_ddp, batch_size, num_workers):
    train_data, train_label, val_data, val_label, num_classes = read_spilt_data(data_path)

    train_dataset = ImgDataSet(train_data, train_label, num_classes)
    val_dataset = ImgDataSet(val_data, val_label, num_classes)

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
    