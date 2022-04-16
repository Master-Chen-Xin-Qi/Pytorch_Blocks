#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         : dataset.py
@Date         : 2022/04/16 16:32:40
@Author       : Xinqi Chen 
@Software     : VScode
@Description  : Dataset and dataloader of spectrum figures
'''

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from config import CONFIG
from utils import label_process


class MyDataset(Dataset):
    def __init__(self, pics_name, labels) -> None:
        super().__init__()
        self.pic_names = pics_name
        self.labels = labels
     
    def __getitem__(self, index):
        pic_file = self.pic_names[index]
        label = self.labels[index]
        pic = Image.open(pic_file).convert('RGB')
        pic = transforms.ToTensor()(pic)
        label = torch.tensor(label, dtype=torch.long)
        return pic, label
    
    def __len__(self):
        return len(self.pic_names)
    
def get_data_loader(pic_names, labels):
    train_val_pics, test_pics, train_val_labels, test_labels = train_test_split(pic_names, labels, test_size=0.1, shuffle=True)
    train_pics, val_pics, train_labels, val_labels = train_test_split(train_val_pics, train_val_labels, test_size=0.2, shuffle=True)
    train_dataset = MyDataset(train_pics, train_labels)
    val_dataset = MyDataset(val_pics, val_labels)
    test_dataset = MyDataset(test_pics, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)
    return train_loader, val_loader, test_loader

# 每个label一个folder，每个folder下是该类别的所有图片，遍历得到所有文件名和label
def get_pic_and_labels(pics_path):
    pic_names = []
    labels = []
    label_dict = label_process(CONFIG["app_name"])
    for root, dirs, _ in os.walk(pics_path):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(root, dir)):
                for file in files:
                    pic_names.append(os.path.join(root, dir, file))
                labels.extend([label_dict[dir]] * len(files))
    return pic_names, labels
    
    
if __name__ == '__main__':
    pics_path = './figs'
    pic_names, labels = get_pic_and_labels(pics_path)
    train_loader, val_loader, test_loader = get_data_loader(pic_names, labels)
    print(f"train_loader: {len(train_loader)} val_loader: {len(val_loader)} \
          test_loader: {len(test_loader)}")