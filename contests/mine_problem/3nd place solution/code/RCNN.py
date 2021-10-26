import os
import glob
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transforms as T
from torchvision.transforms import functional as F
from random import seed, sample, random
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageOps
import plotly.express as px
from bbaug import policies
import RQD


def func_transform():
    res = [T.RandomHorizontalFlip(0.5)]
    return T.Compose(res)


class RDataset(Dataset):

    def __init__(self, transforms, ratio=0.8, train=False):
        seed(5)

        self.train = train
        self.transforms = transforms
        temp = glob.glob(f'data/trainA/*')

        ratio = round(len(temp) * ratio)
        s = sample(temp, k=ratio)

        if train:
            self.dir = s
        else:
            self.dir = list(set(temp) - set(s))
        self.df = pd.read_excel('data/labelA.xlsx')

    def __len__(self):
        return len(self.dir)

    def __getitem__(self, idx):
        # load images and masks

        name = self.dir[idx].split('\\')[1]
        _img = Image.open(self.dir[idx]).convert('RGB')
        img = F.to_tensor(_img)
        # img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # img = img.to(dtype=torch.float).div(255)
        # print(img.dtype)
        # img = F.equalize(img)

        # add boxes
        boxes, labels = [], []
        image_df = self.df[self.df['image_name'] == name]
        for it in zip(image_df['label_name'], image_df['xmin'], image_df['ymin'], image_df['width'],
                      image_df['height']):
            boxes.append([it[1], it[2], it[1] + it[3], it[2] + it[4]])
            if it[0] == 'wood':
                labels.append(1)
            # elif it[0] != 'empty':
            #     labels.append(2)
            else:
                labels.append(2)

        is_crowd = torch.zeros((len(labels),), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': is_crowd}

        # if self.transforms is not None and self.train:
        #     img, target = self.transforms(img, target)

        return img, target


def train():
    from engine import train_one_epoch, evaluate
    import utils

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    dataset_train = RDataset(func_transform(), train=True)
    dataset_test = RDataset(func_transform(), train=False)

    print(len(dataset_train), len(dataset_test))

    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4,
                              collate_fn=utils.collate_fn, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, num_workers=1,
                             collate_fn=utils.collate_fn, drop_last=False)

    model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.91, nesterov=True, weight_decay=0.0001)
    optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-6)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 30
    model.train()
    plot_mem, b_map, b_model = {'loss': [], 'mAP': []}, 0, None
    for epoch in range(num_epochs):
        _, avg_loss = train_one_epoch(model, optimizer, loader_train, torch.device('cuda'), epoch, print_freq=10)
        # lr_scheduler.step()
        evl = evaluate(model, loader_test, device=torch.device('cuda'))
        m_ap = evl.coco_eval['bbox'].stats[0]
        plot_mem['loss'].append(avg_loss)
        plot_mem['mAP'].append(m_ap)

        if m_ap > b_map:
            b_map = m_ap
            b_model = deepcopy(model)

        if (epoch % 10) == 0 and epoch >= 10:
            b_model.cpu().eval()
            torch.save(b_model.state_dict(), f'model_{epoch}.pth')

    print(b_map)
    b_model.cpu().eval()
    torch.save(b_model.state_dict(), 'model.pth')

    px.line(plot_mem, title='training').show()


def test():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load('model_20.pth'))
    model.eval()
    RQD.rqd(model)


if __name__ == '__main__':
    train()
    # test()
