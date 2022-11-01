import pandas as pd
import numpy as np
import lightgbm as lgb
import os 
import re
import argparse
import warnings
warnings.filterwarnings('ignore')
from model.mlp import mlp_eq

def get_arguments():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument()
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

if __name__ == "__main__":
    args = get_arguments()
    args.device = get_device(args)
    
    for area in (0, 1, 2, 3, 4, 5, 6, 7):
        print(f"Area_{area}: ")
        train_data = pd.read_csv(os.path.join('./dataset/processed/AREA_FEATURE', f'area_{area}_train.csv'), index_col=0)
        valid_data = pd.read_csv(os.path.join('./dataset/processed/AREA_FEATURE', f'area_{area}_valid.csv'), index_col=0)
    
    # training dataset
        print("Trainning......")
        # data resample to make balance for label_M=0 and label_M!=0
        long_data = train_data[train_data['label_M']==0]
        short_data = train_data[train_data['label_M']!=0]
        if len(long_data) < len(short_data):
            long_data, short_data = short_data, long_data
        short_data = short_data.sample(len(long_data), replace=True)
        train_data = pd.concat([long_data, short_data]) # len(long_data) == len(short_data)
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        del long_data
        del short_data

        # get feature and label
        target_M = train_data['label_M']
        train_feature = train_data.drop(['label_M', 'label_long', 'label_lati', 'Day'], axis=1) # delete these column

        # discrete magnitude as class
        for i, ss in enumerate(target_M):
            if(ss < 3.5):
                target_M.iloc[i] = 0
            elif(ss < 4.0):
                target_M.iloc[i] = 1
            elif(ss < 4.5):
                target_M.iloc[i] = 2
            elif(ss < 5.0):
                target_M.iloc[i] = 3
            else:
                target_M.iloc[i] = 4
        
        # assign different important weight for each magnitude
        train_data['weight'] = None
        train_data['label_M'] = target_M
        train_data['weight'][train_data['label_M']==0] = 1
        train_data['weight'][train_data['label_M']==1] = 1
        train_data['weight'][train_data['label_M']==2] = 1
        train_data['weight'][train_data['label_M']==3] = 1
        train_data['weight'][train_data['label_M']==4] = 1
        weight_T = train_data['weight'].values
    
    # validation dataset
        valid_M = valid_data['label_M']
        valid_feature = valid_data.drop(['label_M','label_long','label_lati','Day'],axis=1)
        for i, ss in enumerate(valid_M):
            if(ss < 3.5):
                valid_M.iloc[i] = 0
            elif(ss < 4.0):
                valid_M.iloc[i] = 1
            elif(ss < 4.5):
                valid_M.iloc[i] = 2
            elif(ss < 5.0):
                valid_M.iloc[i] = 3
            else:
                valid_M.iloc[i] = 4
        valid_data['weight'] = None
        valid_data['label_M'] = valid_M
        train_data['weight'][train_data['label_M']==0] = 1
        valid_data['weight'][valid_data['label_M']==1] = 1
        valid_data['weight'][valid_data['label_M']==2] = 1
        valid_data['weight'][valid_data['label_M']==3] = 1
        valid_data['weight'][valid_data['label_M']==4] = 1
        weight_V = valid_data['weight'].values

        # manually set different hyper-parameters
        params = {
            'num_leaves': 48,
            'learning_rate': 0.05,
            "boosting": "rf",
            'objective': 'multiclass', # or 'regression'
            'num_class': 5,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.6,
            "bagging_freq": 2,
            "lambda_l1": 0.05,
            "lambda_l2": 0.05,
            "nthread": -1,
            'min_child_samples': 10,
            'max_bin': 200,
            'verbose' : -1
        }
        num_round = 5000
        trn_data = lgb.Dataset(train_feature, label=target_M, weight=weight_T)
        val_data = lgb.Dataset(valid_feature, label=valid_M, weight=weight_V)
        clf = lgb.train(params, trn_data, num_round, valid_sets=[trn_data,val_data], verbose_eval=50, early_stopping_rounds=500)

        ## evaluation
        # oof_lgb = np.matrix(clf.predict(valid_feature, num_iteration=clf.best_iteration))
        # cpf = open(str(area)+'_sm.txt','w+')
        # ccc = oof_lgb.argmax(axis=1)
        # for i in range(len(ccc)):
        #     print(f"{i}, pre:{ccc[i]}, origin:{valid_M[i]}", file=cpf)
        # print((oof_lgb.argmax(axis=1)==(np.matrix(valid_M).T)).sum(),len(oof_lgb))

        # save model for each area
        clf.save_model('./model/'+str(area)+'_mag_model.txt')
