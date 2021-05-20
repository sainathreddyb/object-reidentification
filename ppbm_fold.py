from argparse import ArgumentParser
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
from torch.utils.data import DataLoader
import torchvision.models as models
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torch.nn.functional as F
from skimage import io
from online_triplet_loss.losses import *
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from warmup_scheduler import GradualWarmupScheduler
from models.PPGnet import ClassificationNet
from models.datasets import TigersDatasetTest,TigersDataset
from models.utils import BalancedBatchSampler,LearningRateWarmUP,generate_parts_df
from models.data_transformation import create_transformation
import configparser

def create_black_image(path):
  img = np.zeros((200, 200, 3), np.uint8)
  img=Image.fromarray(img, mode='RGB')
  img.save(path+"zeros.jpg")
  

def main():
  #command line argument parsing
  parser = ArgumentParser()
  parser.add_argument("-f", "--fold",default=0, type=int,
                    help="enter fold")
  args = parser.parse_args()
  fold_index=args.fold
  
  #reading config file
  config = configparser.ConfigParser()
  config.sections()
  config.read('config.ini')  
  
  #
  train_dataset=pd.read_csv(config["train_images"]["fold_directory"]+"fold_train"+str(fold_index)+".csv")
  val_dataset=pd.read_csv(config["train_images"]["fold_directory"]+"fold_test"+str(fold_index)+".csv")
  path=config["train_images"]["parts_dir"]
  create_black_image(path)

  train_dataset=generate_parts_df(train_dataset,path)  
  val_dataset=generate_parts_df(val_dataset,path) 

  train_transform,parts_transform,trunk_transform,val_transform,val_parts_transform,val_trunk_transform=create_transformation()

  target=train_dataset['encoded_labels'].values
  train_images_path=config["train_images"]["train_image_path"]
  part_images_path=config["train_images"]["parts_dir"]
  train_dataset = TigersDataset(train_dataset,train_images_path,part_images_path,train_transform,parts_transform,trunk_transform)
  val_dataset = TigersDataset(val_dataset,train_images_path,part_images_path,val_transform,val_parts_transform,val_trunk_transform)
  
  n_classes = 4
  n_samples = 4
  
  batch_size=n_classes*n_samples
  balanced_batch_sampler_train = BalancedBatchSampler(train_dataset, n_classes, n_samples)
  balanced_batch_sampler_val = BalancedBatchSampler(val_dataset, 8, 2)
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_sampler=balanced_batch_sampler_train)
  validation_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=balanced_batch_sampler_val)
  
  model = ClassificationNet()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), 0.0003)

  # scheduler_warmup is chained with schduler_steplr
  scheduler_steplr = StepLR(optimizer, step_size=80, gamma=0.5)
#   scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=25, after_scheduler=scheduler_steplr)
  scheduler_warmup = LearningRateWarmUP(optimizer=optimizer,
                                   warmup_iteration=0,
                                   target_lr=0.0003,
                                   after_scheduler=scheduler_steplr)

  # this zero gradient update is needed to avoid a warning message, issue #8.
  optimizer.zero_grad()
  optimizer.step()

  n_epochs = 1
  print_every = 10
  margin= 0.3

  valid_loss_min = np.Inf
  val_loss = []
  train_loss = []
  train_acc=[]
  val_acc=[]
  total_step = len(train_loader)
  for epoch in range(1, n_epochs+1):
      running_loss = 0.0
      # scheduler.step(epoch)
      correct = 0
      total=0
      print(f'Epoch {epoch}\n')
      scheduler_warmup.step(epoch)
      for batch_idx, (full_image,part1,part2,part3,part4,part5,part6,body,target_) in enumerate(train_loader):
          
          full_image,part1,part2,part3,part4,part5,part6,body,target_=full_image.to(device),part1.to(device),part2.to(device),part3.to(device),part4.to(device),part5.to(device),part6.to(device),body.to(device),target_.to(device) # on GPU
          # zero the parameter gradients
          optimizer.zero_grad()
          # forward + backward + optimize
          #outputs = model(data_.cuda())
          output1,output2,output3,Zft,Zfl=model(full_image,part1,part2,part3,part4,part5,part6,body)
          
          global_loss = criterion(output1, target_)
          global_trunk_loss= criterion(output2,target_)
          global_part_loss =criterion(output3,target_)
          global_trunk_triplet_loss = batch_hard_triplet_loss(target_, Zft, margin=margin,device=device)
          global_part_triplet_loss = batch_hard_triplet_loss(target_, Zfl, margin=margin,device=device)

          loss=global_loss+ 1.5*global_trunk_loss+1.5* global_part_loss+2*global_trunk_triplet_loss + 2*global_part_triplet_loss

          loss.backward()
          optimizer.step()
          
          # print statistics
          running_loss += loss.item()
          _,pred = torch.max(output1, dim=1)
          correct += torch.sum(pred==target_).item()
          total += target_.size(0)
          if (batch_idx) % print_every == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}' 
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item(),100 * correct/total))
              
      train_acc.append(100 * correct / total)
      train_loss.append(running_loss/total_step)
      print(f'\ntrain loss: {np.mean(train_loss):.4f},\ntrain Acc: {np.mean(train_acc):.4f}')
      batch_loss = 0
      total_t=0
      correct_t=0
      if epoch%5==0:
          print('Evaluation')
          with torch.no_grad():
              model.eval()
              for (full_image,part1,part2,part3,part4,part5,part6,body,target_t) in (validation_loader):
              
                  full_image,part1,part2,part3,part4,part5,part6,body,target_t=full_image.to(device),part1.to(device),part2.to(device),part3.to(device),part4.to(device),part5.to(device),part6.to(device),body.to(device),target_t.to(device) # on GPU
                  output1,output2,output3,Zft,Zfl=model(full_image,part1,part2,part3,part4,part5,part6,body)
          
                  global_loss_t = criterion(output1, target_t)
                  global_trunk_loss_t= criterion(output2,target_t)
                  global_part_loss_t =criterion(output3,target_t)
                  global_trunk_triplet_loss_t = batch_hard_triplet_loss(target_t, Zft, margin=margin,device=device)
                  global_part_triplet_loss_t = batch_hard_triplet_loss(target_t, Zfl, margin=margin,device=device)

                  loss_t=global_loss_t+ 1.5*global_trunk_loss_t+1.5* global_part_loss_t+2*global_trunk_triplet_loss_t + 2*global_part_triplet_loss_t
              
                  batch_loss += loss_t.item()
                  _,pred_t = torch.max(output1, dim=1)
                  correct_t += torch.sum(pred_t==target_t).item()
                  total_t += target_t.size(0)
              val_acc.append(100 * correct_t / total_t)
              val_loss.append(batch_loss/len(validation_loader))
              network_learned = batch_loss < valid_loss_min
              print(f'validation loss: {np.mean(val_loss):.4f}\n val acc: {np.mean(val_acc):.4f}\n')
              # Saving the best weight 
              if network_learned:
                  valid_loss_min = batch_loss
                  torch.save(model.state_dict(),config["train_images"]["train_weights"]+'ppbm_model_fold'+str(fold_index)+'.pt')
                  print('Detected network improvement, saving current model')
      model.train()
  torch.save(model.state_dict(), config["train_images"]["train_weights"]+'ppbm_model_last_fold'+str(fold_index)+'.pt')

if  __name__=="__main__":
  main()











