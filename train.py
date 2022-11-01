import pandas as pd
import numpy as np
import os 
import re
import argparse
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from model.mlp import eqpred_mlp
from torch.utils.data import DataLoader
from dataloader import EqFeaData
from tqdm import tqdm
from utils import get_device, print_model_parameters

def get_arguments():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--input_dim', default=48, type=int, help='the number of used features')
    parser.add_argument('--hidden_dim', default=96, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--saved', default='./saved', type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--gpu_id', default=1, type=int, help='if none gpus, please set -1')
    args = parser.parse_args()
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    
    return args

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)
    
    return model

def train(args, model, train_dataloader, valid_dataloader, area):
    if not os.path.exists(args.saved): os.makedirs(args.saved)
    model_saved = os.path.join(args.saved, f'eqmodel-{area}.pth')
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1.0e-8, weight_decay=5e-4, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in range(1, args.epoch+1):
        model.train()
        total_loss = 0.0
        for batch_idx, (features, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=100):
            features = features.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(batch_idx)
        avg_loss = total_loss / len(train_dataloader)
        
        model.eval()
        with torch.no_grad():
            acc = 0.0
            for batch_idx, (features, labels) in enumerate(valid_dataloader):
                features = features.to(args.device)
                labels = labels.to(args.device)

                output = model(features)
                _, preds = output.max(1) # [B, C]
                acc += preds.eq(labels).sum()
            acc = acc / len(valid_dataloader.dataset) # len(valid_dataloader): the number of batch; len(valid_dataloader.dataset): the number of samples
        
        print('******Epoch {}: Loss: {:.6f} Acc: {:.6f}'.format(epoch, avg_loss, acc))
        if best_acc <= acc:
            torch.save(model.state_dict(), model_saved)

if __name__ == "__main__":
    args = get_arguments()
    args.device = get_device(args)

    for area in (0, 1, 2, 3, 4, 5, 6, 7):
        print(f"==>Area_{area}: ")
        train_data = EqFeaData(area, 'train')
        valid_data = EqFeaData(area, 'valid')
        train_dataloader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
        
        model = eqpred_mlp(args)
        model.double()
        model = model.to(args.device)
        model = init_model(model)

        train(args, model, train_dataloader, valid_dataloader, area)
