import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Resize, ToTensor, ToPILImage, Normalize

class ImgDataset(Dataset):
    def __init__(self, img_path, img_label, num_classes, stroke_path, four_corner_path):
        super().__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.num_classes = num_classes
        self.stroke_path = stroke_path
        self.four_corner_path = four_corner_path
        
        self.transform = transforms.Compose([
            ToPILImage(), 
            Resize(128,128),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, idx):
        img = cv2.imread(self.img_data[idx])
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