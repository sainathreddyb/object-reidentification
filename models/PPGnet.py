#author: sainathb@usf.edu, sneha
#this file contains all 3 networks that are used in the project, PPGNET,
# Cross entropy baseline model and triplet loss baseline model


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


#part pose guided network

class PartNet(nn.Module):
    def __init__(self):
        super(PartNet, self).__init__()   
        ############
        self.part_model = models.resnet34(pretrained=True)

        # for child in self.part_model.children():
        #       for param in child.parameters():
        #           param.requires_grad = False

        self.part_model_base=nn.Sequential(*list(self.part_model.children())[:-3])
        self.part_model_bottom=list(self.part_model.children())[-3]
class ClassificationNet(nn.Module):

    def __init__(self):
        super(ClassificationNet, self).__init__()
        ########### global #####
        self.original_model = models.resnet101(pretrained=True)
        self.full_model = nn.Sequential(*list(self.original_model.children())[:-1])
        self.bn1= nn.BatchNorm1d(2048)
        self.fc1= nn.Linear(2048,107)

        # for child in self.full_model.children():
        #       for param in child.parameters():
        #           param.requires_grad = False
        ############ trunk ###########
        self.fc2 = nn.Linear(2048,107)
        self.bn2= nn.BatchNorm1d(2048)

        self.fc3 = nn.Linear(2048,107)   
        self.bn3= nn.BatchNorm1d(2048)     
        ############ 
        self.body_model = PartNet() 
        self.gap = nn.AdaptiveAvgPool2d((1, 8))

        ############ parts ##########

        self.base1 = PartNet()
        self.base2 = PartNet()
        self.base3 = PartNet()
        self.base4 = PartNet()
        self.base5 = PartNet()
        self.base6 = PartNet()

        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,full_image,part1,part2,part3,part4,part5,part6,body):
        
        ########## global #########
        Dfull = torch.squeeze(self.full_model(full_image))
        output1 = self.fc1(self.bn1(Dfull))

        ########## trunk ##########

        Ftrunk=self.body_model.part_model_base(body)

        Ftrunk= self.gap(Ftrunk)
        part = {}
        #get eight part feature
        for i in range(8):
            part[i] = Ftrunk[:, :, :, i]                    #batch,channel,height,width
        body_feature = torch.cat((part[0], part[1], part[2], part[3], part[4], part[5], part[6], part[7]), dim=1)  #256×8=2048
        Dtrunk = body_feature.view(body_feature.shape[0], -1)

        ######### trunk+full_image #########

        Zft=torch.add(Dtrunk,Dfull)
        output2=self.fc2(self.bn2(Zft))

        ######### parts ############

        feat1 = self.base1.part_model_base(part1)
        feat2 = self.base2.part_model_base(part2)
        feat3 = self.base3.part_model_base(part3)
        feat4 = self.base4.part_model_base(part4)
        feat5 = self.base5.part_model_base(part5)
        feat6 = self.base6.part_model_base(part6)

        feat35 = torch.add(feat3,feat5)
        feat46 = torch.add(feat4,feat6)

        feat1 = self.base1.part_model_bottom(feat1)
        feat1 = self.gap1(feat1)

        feat2 = self.base2.part_model_bottom(feat2)
        feat2 = self.gap2(feat2)

        feat35 = self.base3.part_model_bottom(feat35)
        feat35 = self.gap3(feat35)

        feat46 = self.base4.part_model_bottom(feat46)
        feat46 = self.gap4(feat46)
        
        feature = torch.cat((feat1,feat2,feat35,feat46), dim=1)
        Dlimb = feature.view(feature.shape[0], -1)

        #############parts+full_image#############

        Zfl=torch.add(Dlimb,Dfull)
        output3=self.fc3(self.bn3(Zfl))


        return output1,output2,output3,Zft,Zfl

    def get_embedding(self, x):
        Dfull = torch.squeeze(self.full_model(x))
        # output1 = self.fc1(Dfull)
        return Dfull
        
        
#cross entropy baseline model
        
class CrossEntropyBaseline(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.original_model = models.resnet152(pretrained=True)
        self.model = nn.Sequential(*list(self.original_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2= nn.Linear(1024,107)

        for child in self.model.children():
              for param in child.parameters():
                  param.requires_grad = False

    def forward(self, x):
        output = self.model(x)
        output = self.fc1(torch.squeeze(output))
        output = self.fc2(output)
        return output

    def get_embedding(self, x):
        output = self.model(x)
        output = self.fc1(torch.squeeze(output))
        return output
        
        
#triplet loss baseline model
class TripletLossBaseline(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.original_model = models.resnet152(pretrained=True)
        self.model = nn.Sequential(*list(self.original_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 1024)

        for child in self.model.children():
              for param in child.parameters():
                  param.requires_grad = False

    def forward(self, x):
        output = self.model(x)
        output = self.fc1(torch.squeeze(output))
        return output

    def get_embedding(self, x):
        output = self.model(x)
        output = self.fc1(torch.squeeze(output))
        return output
