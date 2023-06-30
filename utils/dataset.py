import torch
import cv2
import json

from torch.utils.data import Dataset
from torchvision.transforms import transforms, Resize, ToTensor, ToPILImage, Normalize

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class ImgDataset(Dataset):
    def __init__(self, font_path, num_classes, stroke_path, four_corner_path, unicode_path):
        super().__init__()
        self.font_path = font_path
        self.num_classes = num_classes
        self.stroke_path = stroke_path
        self.four_corner_path = four_corner_path
        self.unicode_path = unicode_path
        
        self.transform = transforms.Compose([
            ToPILImage(), 
            Resize(128,128),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def get_stroke(self):
        with open(self.stroke_path, 'r') as f:
            stroke = json.load(f)
        return stroke
    
    def get_four_corner(self):
        with open(self.four_corner_path, 'r') as f:
            four_corner = json.load(f)
        return four_corner
    
    def get_unicode(self, idx):
        with open(self.unicode_path, 'r') as f:
            unicode = f.readlines()
        unicode = [ch.strip() for ch in unicode]
        return unicode[idx]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        label = self.img_label[idx]

        # img 
        img_tensor = self.transform(img)

        # font type
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1.

        # unicode 
        # stroke label
        # four corner

        return img_tensor, label_tensor
    
    def __len__(self):
        return len(self.img_path)