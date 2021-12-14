import torch
import torch.nn as nn
from torch import tensor
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F

import os
import time
import numpy as np
import csv
import pandas as pd
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
batch_size = 32
log_interval = 300

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

target_image_size = (512, 512)
transform=transforms.Compose([
    SquarePad(),
    transforms.Resize(target_image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class SimpleDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.tensor_view = (3, 512, 512)

        self.transforms = transforms
        self.data = []
        self.labels = dataset[:, -1]
        self.label_set = set(self.labels)
  
        for idx, s in enumerate(dataset):
            x = (s[:-1] / 255).view(self.tensor_view)
            y = self.labels[idx].type(torch.LongTensor)
            self.data.append((x, y))

    def __getitem__(self, index):
        if self.transforms != None:
            sample, label = self.data[index]
            sample = self.transforms(sample)
            return sample, label
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)

# Load model

model_ft = models.wide_resnet101_2(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 21)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = SGD(model_ft.parameters(), lr=0.0005, momentum=0.9)
exp_lr_scheduler = StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Load dataset

dataset = datasets.ImageFolder('data/food/train', transform=transform)

train_set, val_set = random_split(dataset, [17000, 552])
trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1)

# Train model

min_loss = float('inf')
for epoch_item in range(epochs):
    print('=== Epoch %d ===' % epoch_item)
    train_loss = 0.
    print(len(trainloader))
    for i, batch in enumerate(trainloader):
        model_ft.train()
        
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer_ft.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()

        train_loss += loss

        if (i+1) % log_interval == 0:
            with torch.no_grad():
                total_val_loss = 0.0
                correct = 0
                total = 0
                
                model_ft.eval()
                for j, data in enumerate(valloader):
                    sample, labels = data
                    sample, labels = sample.to(device), labels.to(device)
                    logits = model_ft(sample)

                    loss = criterion(logits, labels)
                    loss = loss.mean()
                    total_val_loss += loss.item()
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_acc = 100 * correct / total
                total_val_loss /= len(valloader)
                print('=== Epoch: %d/%d, Train Loss: %f, Val Loss: %f, Val Acc: %f' % (
                        epoch_item, i+1,  train_loss/log_interval, total_val_loss, val_acc))
                train_loss = 0.

                if total_val_loss < min_loss:
                    torch.save(model_ft.state_dict(), "data/model_best.pt")
                    min_loss = total_val_loss
                    print("Saving new best model")

    torch.save(model_ft.state_dict(), "data/model_last.pt")
    print("Saving new last model")

# Prediction

f = open('output.csv', 'w')
writer = csv.writer(f)
writer.writerow(['file', 'prediction'])

test_files_path = 'data/food/test'

model_ft.eval()
with torch.no_grad():
    for category_idx, filename in enumerate(os.listdir(test_files_path)):
        image_file = os.path.join(test_files_path, filename)
        img = Image.open(image_file)
        img.load()
        
        trans_img = torch.unsqueeze(transform(img), 0).to(device)
        logits = model_ft(trans_img)
        
        _, predicted = torch.max(logits, 1)
        
        idx_pred = predicted.detach().item()
        cat_pred = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(idx_pred)]
        row = [filename, cat_pred]
        writer.writerow(row)
        
f.close()
