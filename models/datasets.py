import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch.nn.functional as F
import torchvision.models as models
from skimage import io
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import pickle
from torch.utils.data import Dataset

class TigersDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file,root_dir,parts_dir, transform=None,parts_transform=None,trunk_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tigers_file = csv_file
        self.root_dir = root_dir
        self.parts_dir=parts_dir
        self.transform = transform
        self.parts_transform = parts_transform
        self.trunk_transform = trunk_transform

    def __len__(self):
        return len(self.tigers_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.tigers_file.iloc[idx, 1])

        full_image = Image.open(img_name)
        part1=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 3]))
        part2=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 4]))
        part3=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 5]))
        part4=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 6]))
        part5=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 7]))
        part6=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 8]))
        body=Image.open(os.path.join(self.parts_dir,
                                self.tigers_file.iloc[idx, 9]))
 
        ids = self.tigers_file.iloc[idx, 2]
      
        
        if self.transform:
            full_image = self.transform(full_image)

        if self.parts_transform:
            part1=self.parts_transform(part1)
            part2=self.parts_transform(part2)
            part3=self.parts_transform(part3)
            part4=self.parts_transform(part4)
            part5=self.parts_transform(part5)
            part6=self.parts_transform(part6)
        if self.trunk_transform:
            body=self.trunk_transform(body)

        return full_image,part1,part2,part3,part4,part5,part6,body,ids

class TigersDatasetTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file,root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tigers_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.tigers_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.tigers_file.iloc[idx, 0])
        image = Image.open(img_name)     
        names=self.tigers_file.iloc[idx,0]
        
        if self.transform:
            image = self.transform(image)
        return image,names
