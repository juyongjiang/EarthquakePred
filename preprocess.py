import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('ignore')
from pandas.core.indexing import is_label_like
from typing_extensions import final
from tqdm import tqdm
from utils import *
from area_map import area_groups


def merge_em_ga_data(Station_Info_Path:str, Data_Folder_Path:str, Used_features:dict, Merged_Data_Path:dict):
    # Obtain usable stations ID which have MagnUpdate and SoundUpdate
    stationInfo_list = pd.read_csv(Station_Info_Path)
    station_id = stationInfo_list['StationID']
    _continueable_stations = stationInfo_list[stationInfo_list['MagnUpdate']&stationInfo_list['SoundUpdate']]['StationID'].unique()
    _continueable_stations = set(_continueable_stations)
    print(station_id.values, _continueable_stations)
    print("Station ID: ", len(station_id), "==>", len(_continueable_stations))

    EM_GA_Data_Path = os.path.join(Data_Folder_Path, 'raw', 'EM_GA_DATA')
    fileName_list = os.listdir(EM_GA_Data_Path)
    re_magn = re.compile(r'(\d+)_magn.csv')
    re_sound = re.compile(r'(\d+)_sound.csv')
    _set_magn = set()
    _set_sound = set()
    for filename in fileName_list:
        _magn_match = re_magn.findall(filename)
        _sound_match = re_sound.findall(filename)
        if(_magn_match):
            _set_magn.add(int(_magn_match[0])) # station id
            continue
        if(_sound_match):
            _set_sound.add(int(_sound_match[0]))
            continue
    usable_stations = _continueable_stations&_set_magn&_set_sound # the intersection operation on sets

    print('Merge Usable Stations Data:')
    for type in ('magn', 'sound'):
        res = []
        for _id in tqdm(usable_stations, desc=f'{type}:'):
            _df = pd.read_csv(os.path.join(EM_GA_Data_Path, str(_id)+f'_{type}.csv'))[Used_features[type]]
            res.append(_df)
        final_df = pd.concat(res) # dim=0
        final_df.to_pickle(Merged_Data_Path[type])
        del(final_df)


def generate_features(df:pd.DataFrame, window:int, tag:str, start_stamp):
    # df: ['StationID', 'TimeStamp', 'magn@abs_mean'] or ['StationID', 'TimeStamp', 'sound@abs_mean']
    if(len(df)==0): return None
    df.reset_index(drop=True, inplace=True)

    averageName = tag+'@abs_mean' # tag: magn or sound
    df.rename(columns={averageName:'average'}, inplace=True) # rename tag@abs_mean -> average
    df['average'] = df['average'] - df['average'].mean() # normalize data with mean value
    df['diff_1'] = df.groupby('StationID')['average'].shift(1) # group by StationID
    df['diff_1'] = df['average'].values - df['diff_1'].values # generate difference value = current day - last day
    
    # generate day value from timestamp information
    df.loc[:, 'Day'] = df['TimeStamp']
    df['Day'] = df['Day'] - start_stamp # [0, N]
    df['Day'] = (df['Day']//86400 + 1).astype(int) # 24 * 60 * 60
    df.reset_index()
    
    # two DataFrame with difference time granularity, including day and week. 
    tmp = pd.DataFrame(sorted(df['Day'].unique())) # unique operation
    tmp.columns=['Day']
    print("The number of days: ", len(tmp))
    res_df = pd.DataFrame((tmp['Day']//window + 1).unique()).astype(int) # window = 7, denotes a week data
    res_df.columns=['Day']
    res_df['Day'] = res_df['Day']*window # get the end day of each week
    print("The number of weeks: ", len(res_df)) 

    for feature in ['average', 'diff_1']:
        ## (day) feature data
        for tagging in ['max', 'min', 'mean']:
            kk = df.groupby('Day')[feature].agg(tagging) # get the max value, min value, mean value of all local stations in each day. 
            kk.rename(f'{feature}_day_{tagging}', inplace=True)
            tmp = pd.merge(tmp, kk, how='left', on='Day') # merge by Day
        
        ## (week) feature data
        # max_mean, min_mean:
        tmp[f'{feature}_day_max_mean'] = tmp[f'{feature}_day_max'].rolling(window=window, center=False).mean() # window = 7 days => a week
        tmp[f'{feature}_day_min_mean'] = tmp[f'{feature}_day_min'].rolling(window=window, center=False).mean()
        # mean_max, mean_min:
        tmp[f'{feature}_day_mean_max'] = tmp[f'{feature}_day_mean'].rolling(window=window, center=False).max()
        tmp[f'{feature}_day_mean_min'] = tmp[f'{feature}_day_mean'].rolling(window=window, center=False).min()
        
        # merge by day, since we use window=7 in tmp to contruct weekly feature data, the res_df will get the data of each week. 
        res_df = pd.merge(res_df, tmp[['Day', f'{feature}_day_max_mean', f'{feature}_day_min_mean', f'{feature}_day_mean_max', f'{feature}_day_mean_min']], on='Day', how='left')
        # get the mean, max, min, and max-min data in the whole week
        res_df[f'{feature}_mean'] = None
        res_df[f'{feature}_max'] = None
        res_df[f'{feature}_min'] = None
        res_df[f'{feature}_max_min'] = None
        for i, row in res_df.iterrows():
            endDay = row['Day']
            startDay = endDay - window # the length is window
            data_se = df[(df['Day']>startDay)&(df['Day']<=endDay)][feature]
            res_df[f'{feature}_mean'].iloc[i] = data_se.mean() # in fact, it equals tmp[f'{feature}_day_mean'].rolling(window=window, center=False).mean() at the same day
            res_df[f'{feature}_max'].iloc[i] = data_se.max()
            res_df[f'{feature}_min'].iloc[i] = data_se.min()
            res_df[f'{feature}_max_min'].iloc[i] = data_se.max() - data_se.min()

        res_df[f'{feature}_lastday_mean'] = None
        res_df[f'{feature}_lastday_max'] = None
        res_df[f'{feature}_lastday_min'] = None
        res_df[f'{feature}_lastday_max_min'] = None
        for i,row in res_df.iterrows():
            endDay = row['Day']
            data_last = df[df['Day']==endDay][feature]
            res_df[f'{feature}_lastday_mean'].iloc[i] = data_last.mean()
            res_df[f'{feature}_lastday_max'].iloc[i] = data_last.max()
            res_df[f'{feature}_lastday_min'].iloc[i] = data_last.min()
            res_df[f'{feature}_lastday_max_min'].iloc[i] = data_last.max() - data_last.min()
    
    for name in res_df.columns.to_list():
        if(name=='Day'):continue
        res_df.rename(columns={name:(name + '_' + tag)}, inplace=True)
    res_df.dropna(axis=0, how='any', inplace=True) # delete the row with any missing value
    res_df.reset_index(drop=True, inplace=True)
    
    return res_df

def add_label(res_df:pd.DataFrame, eqData_area:pd.DataFrame, start_stamp:int):
    ## adding the label using the earthquake data in the next week
    res_df['label_M'] = None
    res_df['label_long'] = None
    res_df['label_lati'] = None

    zero_stamp = start_stamp
    for i, row in res_df.iterrows():
        endDay = row['Day']
        endStamp = zero_stamp + (endDay-1)*86400
        pre_Range_left = endStamp + 86400*2   
        pre_Range_right = endStamp + 86400*9   # [left, right) becasue we can't get the data on Sunday
        # use historical week's data to predict earthquake happen in the next week and 
        # & use the max magnitude in the local area as the label, which has the max possibility to happen 
        _eq = eqData_area[(eqData_area['Timestamp']<pre_Range_right) & (eqData_area['Timestamp']>=pre_Range_left)]
        if(len(_eq)==0):
            res_df['label_M'].iloc[i] = 0
            res_df['label_long'].iloc[i] = -1
            res_df['label_lati'].iloc[i] = -1
        else:
            _eq_max = _eq.iloc[_eq['Magnitude'].argmax()]
            res_df['label_M'].iloc[i] = _eq_max['Magnitude']
            res_df['label_long'].iloc[i] = _eq_max['Longitude']
            res_df['label_lati'].iloc[i] = _eq_max['Latitude']
    return res_df

    
if __name__ == "__main__":
    ## configuration
    Data_Folder_Path = './dataset/'                     
    Eq_list_path = os.path.join(Data_Folder_Path, 'raw', 'EC_TRAINSET.csv') # earthquake data            
    Station_Info_Path = Data_Folder_Path + 'StationInfo.csv'    # station information data
    Used_features = {'magn':['StationID', 'TimeStamp', 'magn@abs_mean'],
                    'sound':['StationID', 'TimeStamp', 'sound@abs_mean']} # the selected features in EM and GA
    # merge all station's EM and GA data into single file, respectively. 
    Processed_Data_Path = os.path.join(Data_Folder_Path, 'processed')
    if not os.path.exists(Processed_Data_Path): os.makedirs(Processed_Data_Path)
    Area_Feature_Path = os.path.join(Processed_Data_Path, 'AREA_FEATURE') # save constructed features of each area 
    if not os.path.exists(Area_Feature_Path): os.makedirs(Area_Feature_Path)
    Merged_Data_Path= {'magn':os.path.join(Processed_Data_Path, 'magn_data.pkl'),
                       'sound':os.path.join(Processed_Data_Path, 'sound_data.pkl')}
    # split all data into training and validation according to time range
    Time_Range = {'train':['20170101','20220331'],             
                  'valid':['20220401','20220430']}     
                                
    Window = 7  # the size of slide window (/days)                                                                                                 
    
    print("==>Merge Data...")
    if not os.path.exists(Merged_Data_Path['magn']) and not os.path.exists(Merged_Data_Path['sound']):
        merge_em_ga_data(Station_Info_Path, Data_Folder_Path, Used_features, Merged_Data_Path)
        print("Done!")
    else:
        print("Merged Data exist! Skip ...")

    """
        Generate weekly features for each sub-area; 
        Use the max magnitude with corresponding Latitude, and Longitude of the next week as the label in each local area. 
    """
    EqData = pd.read_csv(Eq_list_path)
    magn_data = load_object(Merged_Data_Path['magn'])
    sound_data = load_object(Merged_Data_Path['sound'])

    if not os.listdir(Area_Feature_Path):
        for i, area in enumerate(area_groups):
            print(f"==>Area: {i}")
            ID_list = area['id'] # id set
            range_list = area['range'] # latitude and longitude
            # extract earthquake data in each area as label
            eqData_area = EqData[(EqData['Latitude']>=range_list[0]) & (EqData['Latitude']<=range_list[1]) & 
                                    (EqData['Longitude']>=range_list[2]) & (EqData['Longitude']<=range_list[3])]

            # area data with ID_list
            local_magn_data = magn_data[magn_data['StationID'].apply(lambda x:x in ID_list)].reset_index(drop=True)
            local_sound_data = sound_data[sound_data['StationID'].apply(lambda x:x in ID_list)].reset_index(drop=True)
            for flag in ['train', 'valid']:
                time_range = Time_Range[flag]
                start_stamp = string2stamp(time_range[0])
                end_stamp = string2stamp(time_range[1])

                # extract data in specific time range
                _df_magn = local_magn_data[(local_magn_data['TimeStamp']>=start_stamp)&(local_magn_data['TimeStamp']<end_stamp)]
                _df_sound = local_sound_data[(local_sound_data['TimeStamp']>=start_stamp)&(local_sound_data['TimeStamp']<end_stamp)]
                
                # construct new features with magn@abs_mean and sound@abs_mean
                # ['StationID', 'TimeStamp', 'magn@abs_mean'] or ['StationID', 'TimeStamp', 'sound@abs_mean']
                print("------", flag, '@magn')
                _magn_res = generate_features(_df_magn, Window, 'magn', start_stamp)
                _magn_res = add_label(_magn_res, eqData_area, start_stamp)
                print("------", flag, '@sound')
                _sound_res = generate_features(_df_sound, Window, 'sound', start_stamp)
                _sound_res = add_label(_sound_res, eqData_area, start_stamp)

                # drop surplus label column
                _magn_res.drop(['label_M', 'label_long', 'label_lati'], axis=1, inplace=True) # delete columns of ['label_M', 'label_long', 'label_lati']
                _final_res = pd.merge(_magn_res, _sound_res, on='Day', how='left')
                _final_res.dropna(inplace=True)
                _final_res.to_csv(os.path.join(Area_Feature_Path, f'area_{i}_{flag}.csv'))
    else:
        print("Area features files exist! Skip ...")

