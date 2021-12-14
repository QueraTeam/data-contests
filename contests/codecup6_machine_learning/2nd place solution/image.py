#!/usr/bin/env python
# coding: utf-8

# ## Imports


import os
import gc
import cv2
import glob

import numpy as np
import pandas as pd
from pathlib import Path
Path.ls = lambda x: list(x.iterdir())

import torch
from torch import nn
import torch.nn.functional as F

import timm
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


# ## EDA

train_path = "../input/code-cup-food-dataset/food/train"
test_path = "../input/code-cup-food-dataset/food/test"


train_image_files = glob.glob(train_path + "/**/*")
test_image_files = glob.glob(test_path + "/*")


df = pd.DataFrame({
    'images': train_image_files
})
df['label_txt'] = df['images'].map(lambda x: x.split("/")[5])
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label_txt'])


test_df = pd.DataFrame({
    'images': test_image_files
})


print("Unique Labels: ", df['label_txt'].nunique())



# ## Dataset and DataLoader

def get_transforms(mean, std, size):
    
    train_transforms = A.Compose([
        A.Resize(size, size),
#         A.RandomResizedCrop(size, size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Normalize(mean=mean, std=std)
    ])
    
    valid_transforms = A.Compose([
        A.Resize(size, size),
#         A.CenterCrop(size, size),
        A.Normalize(mean=mean, std=std)
    ])
    
    return train_transforms, valid_transforms


 


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.paths = df['images'].values
        self.labels = df['label'].values
        self.transforms = transforms
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        image = torch.tensor(image).float().permute(2, 0, 1)
        label = torch.tensor(label).long()

        return image, label

    def __len__(self):
        return len(self.df)


 


def denormalize(image, mean, std):

    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    return image * std + mean


 


def visualize_batch(batch, mean=None, std=None):
    images, targets = batch

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    for i, (image, target) in enumerate(zip(images[:16], targets[:16])):
        image = image.permute(1, 2, 0)
        image = denormalize(image, mean=mean, std=std)
        
        axes[i].imshow(image)
        axes[i].set_title(target.item())
        axes[i].axis("off")

    plt.show()


 


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# ## Model


class Model(nn.Module):
    def __init__(self, model_name, pretrained, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, 
                                       pretrained=pretrained, 
                                       num_classes=num_classes)
        
        
    def forward(self, x):
        x = self.model(x)
        return x


# ## Training Functions

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        self.avg, self.sum, self.count = [0]*3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def get_per_class_accuracy(preds, targets, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix.diag() / confusion_matrix.sum(1)

def get_accuracy(preds, targets):
    preds = preds.argmax(dim=-1)
    return (preds == targets).float().mean()

def train_one_epoch(model, 
                    criterion, 
                    train_loader,
                    optimizer=None, 
                    lr_scheduler=None):
    
    model.train()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for images, targets in tqdm_object:
        images, targets = images.to(CFG.device), targets.to(CFG.device)
        
        preds = model(images)
        loss = criterion(preds, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if CFG.step == "batch":
            lr_scheduler.step()
                
        count = images.size(0)
        loss_meter.update(loss.item(), count)
        
        accuracy = get_accuracy(preds.detach(), targets)
        acc_meter.update(accuracy.item(), count)

        lr = get_lr(optimizer)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, 
                                train_accuracy=acc_meter.avg, 
                                lr=lr)
    
    return loss_meter, acc_meter


def valid_one_epoch(model, 
                    criterion, 
                    valid_loader):
    
    model.eval()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for images, targets in tqdm_object:
            images, targets = images.to(CFG.device), targets.to(CFG.device)
            
            preds = model(images)
            loss = criterion(preds, targets)
            
            count = images.size(0)
            loss_meter.update(loss.item(), count)
            
            accuracy = get_accuracy(preds.detach(), targets)
            acc_meter.update(accuracy.item(), count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg, 
                                    valid_accuracy=acc_meter.avg)
    
    return loss_meter, acc_meter


def train_eval(model, 
               train_loader,
               valid_loader,
               criterion, 
               optimizer, 
               lr_scheduler):
    
    best_loss = float('inf')
    best_score = -float('inf')
    
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}")
        current_lr = get_lr(optimizer)
        
        train_loss, train_acc = train_one_epoch(model, 
                                                criterion, 
                                                train_loader,
                                                optimizer=optimizer,
                                                lr_scheduler=lr_scheduler)
        
        valid_loss, valid_acc = valid_one_epoch(model,
                                                criterion,
                                                valid_loader)
        
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f'best_loss_{CFG.fold}.pt')
            print("Saved best loss model!")
        
        if valid_acc.avg > best_score:
            best_score = valid_acc.avg
            torch.save(model.state_dict(), f"best_score_{CFG.fold}.pt")
            print("Saved best scoring model!")

        
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(valid_acc.avg)

def main(train_df, valid_df, logger=None):
    gc.collect()
    train_transforms, valid_transforms = get_transforms(mean=CFG.mean, 
                                                        std=CFG.std,
                                                        size=CFG.size)
    
    train_dataset = Dataset(train_df, transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=CFG.batch_size,
                                               shuffle=True,
                                               num_workers=CFG.num_wrokers)
    
    
    valid_dataset = Dataset(valid_df, transforms=valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=CFG.batch_size,
                                               shuffle=False,
                                               num_workers=CFG.num_wrokers)
    
    
    
    model = Model(model_name=CFG.model_name,
                  pretrained=CFG.pretrained,
                  num_classes=CFG.num_classes)
    
    model.to(CFG.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    
    total_training_steps = len(train_loader) * CFG.epochs
    
    if CFG.lr_scheduler == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  patience=CFG.patience, 
                                                                  factor=CFG.factor,
                                                                  min_lr=1e-6,
                                                                  mode='max')
    elif CFG.lr_scheduler == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=CFG.lr,
                                                           epochs=CFG.epochs,
                                                           steps_per_epoch=len(train_loader),
                                                           div_factor=25.0,
                                                           pct_start=CFG.pct_start)
    elif CFG.lr_scheduler == 'Cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                       num_warmup_steps=int(0.1 * total_training_steps),
                                                       num_training_steps=total_training_steps)
    
    
    train_eval(model, 
               train_loader,
               valid_loader,
               criterion, 
               optimizer, 
               lr_scheduler)


class CFG:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    model_name = 'tf_efficientnet_b3_ns'
    pretrained = True
    num_classes = 21
    
    size = 360
    batch_size = 32
    num_wrokers = 4
    
    epochs = 12
    lr = 1e-3
    lr_scheduler = 'OneCycleLR'
    step = 'batch'
    pct_start = 0.15
    patience = 2
    factor = 0.8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for i, (_, val_idx) in enumerate(kfold.split(df, df['label'])):
    df.loc[val_idx, 'fold'] = i


for fold in range(5):
    CFG.fold = fold
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    print(train_df.shape, valid_df.shape)

    main(train_df, valid_df)


# ## Inference


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.paths = df['images'].values
        self.transforms = transforms
    
    def __getitem__(self, idx):
        path = self.paths[idx]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        image = torch.tensor(image).float().permute(2, 0, 1)

        return image

    def __len__(self):
        return len(self.df)


 


_, valid_transforms = get_transforms(mean=CFG.mean, 
                                    std=CFG.std,
                                    size=CFG.size)

test_dataset = TestDataset(test_df, valid_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=CFG.batch_size,
                                           shuffle=False,
                                           num_workers=CFG.num_wrokers)


 


def evaluate(model_path, test_loader):
    model = Model(model_name=CFG.model_name,
                  pretrained=CFG.pretrained,
                  num_classes=CFG.num_classes)
    
    model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    all_preds = []
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for images in tqdm_object:
            images = images.to(CFG.device)
            preds = model(images)
            all_preds.append(preds)
            
    return torch.cat(all_preds).cpu()


 


fold_preds = []
for i in range(5):
    preds = evaluate(f'best_score_{i}.pt', test_loader)
    fold_preds.append(preds)


 


predictions = torch.stack(fold_preds).mean(dim=0).argmax(dim=-1).numpy()


 


test_df['file'] = test_df['images'].map(lambda x: x.split("/")[-1])
test_df['prediction'] = label_encoder.inverse_transform(predictions)
result = test_df.drop('images', axis=1)

result.to_csv("output.csv", index=False)

