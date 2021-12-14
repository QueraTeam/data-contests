
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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm


# ## EDA

df = pd.read_csv("../input/code-cup-ratings/data/train.csv")
test_df = pd.read_csv("../input/code-cup-ratings/data/test.csv")


user_id_encoder = LabelEncoder().fit(df.userId)
item_id_encoder = LabelEncoder().fit(df.itemId)

df['userId'] = user_id_encoder.transform(df['userId'])
df['itemId'] = item_id_encoder.transform(df['itemId'])

test_df['userId'] = user_id_encoder.transform(test_df['userId'])
test_df['itemId'] = item_id_encoder.transform(test_df['itemId'])


# ## Dataset and DataLoader


def build_loaders(df, fold, batch_size):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)
    
    train_df_tensor = torch.tensor(train_df[['userId', 'itemId', 'rating']].values).long()
    valid_df_tensor = torch.tensor(valid_df[['userId', 'itemId', 'rating']].values).long()
    
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_df_tensor[:, :2], train_df_tensor[:, -1]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(valid_df_tensor[:, :2], valid_df_tensor[:, -1]),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader





kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for i, (_, val_idx) in enumerate(kfold.split(df, df['rating'])):
    df.loc[val_idx, 'fold'] = i


# ## Model


class Model(nn.Module):
    def __init__(
        self, 
        num_users, 
        num_items, 
        embed_dim, 
        ranged_sigmoid=False
    ):
        super().__init__()
        self.user = nn.Embedding(num_users, embed_dim)
        self.item = nn.Embedding(num_items, embed_dim)
        self.layers = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.ranged_sigmoid = ranged_sigmoid
        
    def forward(self, x):
        user_embed = self.user(x[:, 0])
        item_embed = self.item(x[:, 1])
        concat = torch.cat([user_embed, item_embed], dim=1)
        preds = self.layers(concat)
        sigmoid = torch.sigmoid(preds)
        if self.ranged_sigmoid:
            sigmoid = sigmoid * 3 + 1
        
        return sigmoid.view(-1)


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





def train_one_epoch(model, 
                    criterion, 
                    train_loader,
                    optimizer=None, 
                    lr_scheduler=None):
    
    model.train()
    loss_meter = AvgMeter()
    rmse_meter = AvgMeter()
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for ids, targets in tqdm_object:
        ids, targets = ids.to(CFG.device), targets.to(CFG.device)
        
        preds = model(ids)
        loss = criterion(preds, targets.float())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if CFG.step == "batch":
            lr_scheduler.step()
                
        count = ids.size(0)
        loss_meter.update(loss.item(), count)
        
        rmse = nn.MSELoss()(preds, targets.float()).sqrt()
        rmse_meter.update(rmse.item(), count)

        lr = get_lr(optimizer)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, 
                                train_rmse=rmse_meter.avg, 
                                lr=lr)
    
    return loss_meter, rmse_meter





def valid_one_epoch(model, 
                    criterion, 
                    valid_loader):
    
    model.eval()
    loss_meter = AvgMeter()
    rmse_meter = AvgMeter()
    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    with torch.no_grad():
        for ids, targets in tqdm_object:
            ids, targets = ids.to(CFG.device), targets.to(CFG.device)
            
            preds = model(ids)
            loss = criterion(preds, targets.float())
            
            count = ids.size(0)
            loss_meter.update(loss.item(), count)
            
            rmse = nn.MSELoss()(preds, targets.float()).sqrt()
            rmse_meter.update(rmse.item(), count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg, 
                                    valid_rmse=rmse_meter.avg)
    
    return loss_meter, rmse_meter





def train_eval(model, 
               train_loader,
               valid_loader,
               criterion, 
               optimizer, 
               lr_scheduler):
    
    best_loss = float('inf')
    best_score = float('inf')
    
    for epoch in range(CFG.epochs):
        print(f"Epoch {epoch + 1}")
        current_lr = get_lr(optimizer)
        
        train_loss, train_rmse = train_one_epoch(model, 
                                                criterion, 
                                                train_loader,
                                                optimizer=optimizer,
                                                lr_scheduler=lr_scheduler)
        
        valid_loss, valid_rmse = valid_one_epoch(model,
                                                criterion,
                                                valid_loader)
        
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f'best_loss_{CFG.fold}.pt')
            print("Saved best loss model!")
        
        if valid_rmse.avg < best_score:
            best_score = valid_rmse.avg
            torch.save(model.state_dict(), f"best_score_{CFG.fold}.pt")
            print("Saved best scoring model!")

        
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(valid_rmse.avg)





def main(df, logger=None):
    gc.collect()
    train_loader, valid_loader = build_loaders(df, CFG.fold, CFG.batch_size)
    
    
    model = Model(num_users=df['userId'].nunique(),
                  num_items=df['itemId'].nunique(),
                  embed_dim=CFG.embed_dim,
                  ranged_sigmoid=True)
    
    model.to(CFG.device)
    criterion = nn.MSELoss()
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
    batch_size = 2048
    
    embed_dim = 128
    
    epochs = 10
    lr = 1e-2
    lr_scheduler = 'ReduceLROnPlateau'
    step = 'epoch'
    pct_start = 0.15
    patience = 1
    factor = 0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





for fold in range(5):
    CFG.fold = fold
    main(df)



test_df_tensor = torch.tensor(test_df[['userId', 'itemId']].values).long()


test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_df_tensor),
    batch_size=256,
    shuffle=False,
)



def evaluate(df, model_path, test_loader):
    model = Model(num_users=df['userId'].nunique(),
                  num_items=df['itemId'].nunique(),
                  embed_dim=CFG.embed_dim,
                  ranged_sigmoid=True)
    
    model.to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    all_preds = []
    tqdm_object = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for ids in tqdm_object:
            ids = ids[0].to(CFG.device)
            preds = model(ids)
            all_preds.append(preds)
            
    return torch.cat(all_preds).cpu()





fold_preds = []
for fold in range(5):
    preds = evaluate(df, f"best_score_{fold}.pt", test_loader)
    fold_preds.append(preds)



preds = torch.stack(fold_preds).mean(dim=0)



result = pd.DataFrame(
    {'prediction': preds}
)
result.to_csv("output.csv", index=False)



















