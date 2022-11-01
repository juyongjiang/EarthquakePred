import os
import random
import torch
import numpy as np
from time import strptime, mktime, localtime, strftime


def set_seed(args):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)   
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def get_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
    else:
        device = torch.device("cpu")
    set_seed(args) # reproductibility

    return device

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def dump_object(path:str, target_obj):
    from pickle import dump
    with open(path, 'wb') as ff:
        dump(target_obj, ff)

def load_object(path:str):
    from pickle import load
    with open(path, 'rb') as ff:
        s = load(ff)
    return s

# string <-> stamp
def string2stamp(timeString):
    _timeTuple = strptime(timeString, "%Y%m%d")
    return int(mktime(_timeTuple))

def stamp2string(timeStamp):
    _timeTuple = localtime(float(timeStamp))
    return strftime("%Y%m%d", _timeTuple)


def reduce_mem(df):
    import numpy as np
    import gc
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df