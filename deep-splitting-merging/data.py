import numpy as np
import cv2
import torch.utils.data as utils
from PIL import Image, ImageChops 
import torch
from torch import optim
import PIL.ImageOps
from torch.utils.data import Dataset, DataLoader  
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import xml.etree.ElementTree as ET
from typing import Tuple, Union
import math
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import json
from pathlib import Path

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
        rows = data['rows']
        columns = data['columns']
    
    return rows, columns

def resize_img(img):
    #img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = np.array(img)
    h, w = img.shape
    min_width = 100
    scale = 0.5
    new_h = int(scale * h) if int(scale * h) > min_width else min_width
    new_w = int(scale * w) if int(scale * w) > min_width else min_width
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, new_h, new_w

def resize_labels(row_labels, col_labels, new_h, new_w):
    width = int(np.floor(new_w / 8))
    height = int(np.floor(new_h / 8))

    row_label = np.array([row_labels]).T * np.ones((len(row_labels), width))
    column_label = np.array(col_labels) * np.ones((height, len(col_labels)))

    row_label = np.array(row_label, dtype=np.uint8)
    column_label = np.array(column_label, dtype=np.uint8)

    row_label = cv2.resize(row_label, (width, new_h))
    column_label = cv2.resize(column_label, (new_w, height))

    row_label = row_label[:, 0]
    column_label = column_label[0, :]
    
    return row_label, column_label

class TTruthDataset(Dataset):
    def __init__(self, root_dir, images_dir, json_dir, transform):
        self.root_dir = root_dir
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.transform = transform
        
    def __getitem__(self, index):
        img_path = self.images_dir + '/' + str(index + 1) + '.jpg'
        json_path = self.json_dir + '/' + str(index + 1) + '.json'
        img = Image.open(img_path)
        row_labels, col_labels = load_labels(json_path)
        width, height = img.size
        
        if width > 1800 or height > 1800:
            img, new_h, new_w = resize_img(img)
            row_labels, col_labels = resize_labels(row_labels, col_labels, new_h, new_w)
        
        if self.transform is not None:
            img = self.transform(img)
        row_labels = torch.FloatTensor(row_labels)
        col_labels = torch.FloatTensor(col_labels)

        return img, row_labels, col_labels, index + 1
    
    def __len__(self):
        return len(self.root_dir)

def data_loaders(root_path, images_path, json_path):
    root_dir = root_path
    images_dir = images_path
    json_dir = json_path
    
    root_dir = datasets.ImageFolder(root = root_dir)
    ttruth_dataset = TTruthDataset(root_dir, images_dir, json_dir, transform = transforms.ToTensor())
    
    dataset_size = len(ttruth_dataset)
    indices = list(range(dataset_size))
    training_split = int(0.8 * dataset_size)
    validation_split = int(0.9 * dataset_size)

    np.random.seed(96)
    np.random.shuffle(indices)

    train_indices = indices[:training_split]
    valid_indices = indices[training_split:validation_split]
    test_indices = indices[validation_split:]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    training_loader = DataLoader(ttruth_dataset,
                        num_workers = 0,
                        batch_size = 1,
                        sampler = train_sampler)

    validation_loader = DataLoader(ttruth_dataset,
                        num_workers = 0,
                        batch_size = 1,
                        sampler = valid_sampler)

    testing_loader = DataLoader(ttruth_dataset,
                        num_workers = 0,
                        batch_size = 1,
                        sampler= test_sampler)
    
    return training_loader, validation_loader, testing_loader