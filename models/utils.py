
import pandas as pd
import numpy as np
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch.nn.functional as F
from skimage import io
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from online_triplet_loss.losses import *
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from warmup_scheduler import GradualWarmupScheduler

def generate_parts_df(df,path):
  #creating a data frame with parts
  parts_list=[]
  for image_id in df["image"].values:
    image_id=image_id.split(".")[0]
    
    parts={"1":"","2":"","3":"","4":"","5":"","6":"","body":""}
    for i in range(1,7):
      image_name=image_id+"_"+str(i)+"part.jpg"
      image_path=path+image_name
      if os.path.exists(image_path):
        parts[str(i)]=image_name
      else:
        parts[str(i)]="zeros.jpg"

    image_name=image_id+"_body.jpg"
    image_path=path+image_name
    if os.path.exists(image_path):
      parts["body"]=image_name
    else:
      parts["body"]="zeros.jpg"
    parts_list.append(parts)
  parts_df=pd.DataFrame(parts_list)
  parts_df = parts_df.reset_index(drop=True)
  df=pd.concat([df,parts_df],axis=1)
  df = df.reset_index(drop=True)
  return df


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _,_,_,_,_,_,_,_,label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes


    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
        
class LearningRateWarmUP(nn.Module):
    #learning rate sheduler
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr*float(cur_iteration)/float(self.warmup_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration-self.warmup_iteration)

