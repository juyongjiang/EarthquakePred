import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EqFeaData(Dataset):
    def __init__(self, area, flag): # flag = train or valid
        super(EqFeaData, self).__init__()
        feature_label_data = pd.read_csv(os.path.join('./dataset/processed/AREA_FEATURE', f'area_{area}_{flag}.csv'), index_col=0)

        # data resample to make balance for label_M=0 and label_M!=0
        if flag == 'train':
            long_data = feature_label_data[feature_label_data['label_M']==0]
            short_data = feature_label_data[feature_label_data['label_M']!=0]
            if len(long_data) < len(short_data):
                long_data, short_data = short_data, long_data
            short_data = short_data.sample(len(long_data), replace=True)
            feature_label_data = pd.concat([long_data, short_data]) # len(long_data) == len(short_data)
            feature_label_data = feature_label_data.sample(frac=1).reset_index(drop=True)
            del long_data
            del short_data

        # get feature and label
        self.target_M = feature_label_data['label_M']
        self.target_M = self.discrete_label(self.target_M)  # discrete magnitude as class
        self.target_M = np.array(self.target_M.values) # transfer to numpy format
        self.feature = feature_label_data.drop(['label_M', 'label_long', 'label_lati', 'Day'], axis=1) # delete these column
        self.feature = np.array(self.feature.values)

    def discrete_label(self, label_M):
        for i, ss in enumerate(label_M):
            if(ss < 3.5):
                label_M.iloc[i] = 0
            elif(ss < 4.0):
                label_M.iloc[i] = 1
            elif(ss < 4.5):
                label_M.iloc[i] = 2
            elif(ss < 5.0):
                label_M.iloc[i] = 3
            else:
                label_M.iloc[i] = 4
        return label_M
    
    def __len__(self):
        return self.feature.shape[0] # [N, D]
    
    def __getitem__(self, idx):
        feature_row = self.feature[idx]
        label_row = self.target_M[idx]

        return torch.tensor(feature_row, dtype=torch.float64), torch.tensor(label_row, dtype=torch.long)