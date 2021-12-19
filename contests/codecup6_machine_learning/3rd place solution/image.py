# -*- coding: utf-8 -*-
"""4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GjHaboV7Wwago3L5pFbRA6y_C0HWxIvS
"""

# !wget http://156.253.5.172/food.zip

# !unzip food.zip

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import shutil

#Add Additional libraries here
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

random.seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
])

class CatDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.image_list = []


        for cls_index in range(len(self.classes)):

            class_files = [[os.path.join(self.root_dir, self.classes[cls_index], x), cls_index] for x in os.listdir(os.path.join(self.root_dir, self.classes[cls_index]))]
            self.image_list += class_files
                
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]
        
        # Read the image
        image = Image.open(path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img':   image,
                'label': label,
                'paths' : path}

        return data

batchsize = 32

classes = os.listdir('food/train')

dataset = CatDataset(root_dir='food/train/',
                          classes = classes,
                          transform= train_transformer)
print()
N = len(dataset)
temp = 1000
train_N = int((N - temp) * 0.8)
val_N = N - train_N - temp
print(N, train_N, val_N, temp)
train_set, val_set, _ = torch.utils.data.random_split(dataset, [train_N, val_N, temp])

train_loader = DataLoader(train_set, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batchsize, drop_last=False, shuffle=False)

import torchvision.models as models
model = models.densenet121(pretrained=True).cuda()
modelname = 'densenet121'

model.fc = nn.Linear(4096, len(classes)).cuda()

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_epoch = 10

def train(optimizer, epoch):
    
    model.train()
    
    train_loss = 0
    train_correct = 0
    
    INF = 16
    
    for batch_index, batch_samples in enumerate(train_loader):
        data, target = batch_samples['img'].cuda(), batch_samples['label'].cuda()
   
        optimizer.zero_grad()
        output = model(data).cuda()
        
        criteria = nn.CrossEntropyLoss()
        loss = criteria(output, target.long())
        train_loss += criteria(output, target.long()) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()
        if batch_index % INF == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, len(train_loader),
                100.0 * batch_index / len(train_loader), loss.item()/ INF))
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss/len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    
def val(epoch):
    
    model.eval()
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    
    criteria = nn.CrossEntropyLoss()
    val_loss = 0
    correct = 0
    with torch.no_grad():

        predictions = []
        scores = []
        targets =[]
        for batch_index, batch_samples in enumerate(val_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)
            
            val_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            _target = target.long().cpu().numpy()
            predictions = np.append(predictions, pred.cpu().numpy())
            scores = np.append(scores, score.cpu().numpy()[:,1])
            targets = np.append(targets,_target)
           
          
    return targets, scores, predictions

def test():
    
    model.eval()
    
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    
    
    criteria = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        predictions = []
        scores = []
        targets =[]
        for batch_index, batch_samples in enumerate(test_loader):
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)
            
            test_loss += criteria(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()

            _target = target.long().cpu().numpy()
            predictions = np.append(predictions, pred.cpu().numpy())
            scores = np.append(scores, score.cpu().numpy()[:,1])
            targets = np.append(targets,_target)
           
          
    return targets, scores, predictions
vote_pred = np.zeros(val_set.__len__())
vote_score = np.zeros(val_set.__len__())
C = 1

def calc_metrics(targetlist, scorelist, predlist):
    vote_pred = predlist 
    vote_score = scorelist 
    TP = ((vote_pred == 1) & (targetlist == 1)).sum()
    TN = ((vote_pred == 0) & (targetlist == 0)).sum()
    FN = ((vote_pred == 0) & (targetlist == 1)).sum()
    FP = ((vote_pred == 1) & (targetlist == 0)).sum()


    print('TP=',TP,'TN=',TN,'FN=',FN,'FP=',FP)
    p = TP / (TP + FP)
    print('precision', p)
    r = TP / (TP + FN)
    print('recall',r)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('F1',F1)
    print('acc',acc)
            
    vote_pred = np.zeros(val_set.__len__())
    vote_score = np.zeros(val_set.__len__())
    print('\nThe epoch is {}, recall: {:.4f}, precision: {:.4f}, F1: {:.4f}, accuracy: {:.4f}'.format(
    epoch, r, p, F1, acc ))
    
for epoch in range(1, total_epoch+1):
    train(optimizer, epoch)
    
    targetlist, scorelist, predlist = val(epoch)
    calc_metrics(targetlist, scorelist, predlist)

    if epoch % C == 0:
        torch.save(model.state_dict(), "{}.pt".format(modelname))

class TestDataset(Dataset):
    def __init__(self, root_dir, classes=['None'], transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.image_list = []


        for cls_index in range(len(self.classes)):

            class_files = [[os.path.join(self.root_dir, x), cls_index] for x in os.listdir(root_dir)]
            self.image_list += class_files
                
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]
        
        # Read the image
        image = Image.open(path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img':   image,
                'label': label,
                'paths' : path}

        return data

test_set = TestDataset(root_dir='food/test/',
                          transform= val_transformer)
test_loader = DataLoader(test_set, batch_size=batchsize, drop_last=False, shuffle=False)

predictions = []
with torch.no_grad():
  
  for batch_index, batch_samples in enumerate(test_loader):
    data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
    output = model(data)
    score = F.softmax(output, dim=1)
    pred = output.argmax(dim=1, keepdim=True)
    for item in pred:
      predictions.append(item[0].cpu().numpy())

labels = list(map(lambda x: classes[x], predictions))
import pandas as pd
names = []
for i, image_name in enumerate(os.listdir("food/test")):
  names.append(image_name)

pd.DataFrame({"file": names, "prediction": labels}).to_csv("output.csv", index=False)