# import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')
from utils import *
from numpy.lib.function_base import extract
from preprocess import generate_features


if __name__ == "__main__":
	zip_path = './dataset/AETA_20220529-20220702/'
	save_path = './dataset/data_week/'
	if not os.path.exists(save_path): os.makedirs(save_path)
	time_range = ['20220529', '20220604'] # a week range
	file_name = 'EM&GA_' + time_range[0] + '-' + time_range[1] # time1 - time2
	extractpath = os.path.join(save_path, file_name) # save unzip files
	if not os.path.exists(extractpath):
		frzip = zipfile.ZipFile(os.path.join(zip_path, file_name+'.zip'), 'r')
		frzip.extractall(extractpath)
		frzip.close()
	else:
		print("Already unzip. Skip ...")
	
	# check whether there exists .zip files
	em_path = os.path.join(f'{extractpath}', f'EM_{time_range[0]}-{time_range[1]}')
	ga_path = os.path.join(f'{extractpath}', f'GA_{time_range[0]}-{time_range[1]}')
	print(em_path, ga_path)
	em_id, ga_id = set(), set()
	for filename in os.listdir(em_path):
		if(filename.endswith('.csv')):
			em_id.add(eval(filename.split('_')[0]))
			continue
		with zipfile.ZipFile(em_path+filename, 'r') as frzip:
			frzip.extractall(em_path)
	for filename in os.listdir(ga_path):
		if(filename.endswith('.csv')):
			ga_id.add(eval(filename.split('_')[0]))
			continue
		with zipfile.ZipFile(ga_path+filename, 'r') as frzip:
			frzip.extractall(ga_path)

	# start prediction
	area_groups = [
        {'id':set([133, 246, 119, 122, 59, 127]),'range':[30,34,98,101]},
        {'id':set([128, 129, 19, 26, 159, 167, 170, 182, 310, 184, 188, 189, 191, 197, 201, 204, 88, 90, 91, 93, 94, 221, 223, 98, 107, 235, 236, 252, 250, 124, 125]),'range':[30,34,101,104]},
        {'id':set([141, 150, 166, 169, 43, 172, 183, 198, 202, 60241, 212, 214, 99, 228, 238, 115, 116, 121, 251]),'range':[30,34,104,107]},
        {'id':set([131, 36, 164, 165, 231, 60139, 174, 175, 206, 303, 82, 51, 243, 55, 308, 119, 313, 318]),'range':[26,30,98,101]},
        {'id':set([256, 130, 132, 147, 148, 149, 151, 153, 32, 33, 35, 60195, 38, 39, 41, 302, 304, 177, 305, 307, 181, 309, 314, 315, 316, 317, 319, 320, 193, 322, 200, 73, 329, 75, 333, 78, 334, 84, 87, 60251, 96, 225, 101, 229, 105, 109, 40047, 240, 247, 120, 254, 255]),'range':[26,30,101,104]},
        {'id':set([352, 321, 355, 324, 326, 328, 331, 77, 47, 48, 335, 339]),'range':[26,30,104,107]},
        {'id':set([161, 226, 137, 138, 171, 140, 113, 306, 152, 186, 220, 60157]),'range':[22,26,98,101]},
        {'id':set([50117, 327, 106, 332, 142, 146, 24, 155, 156, 29]),'range':[22,26,101,104]}
    ]	
	max_mag = -1
	eq_area = -1
	print(em_id, ga_id)
	for area in (0, 1, 2, 3, 4, 5, 6, 7):
		## concat all EM and GA data from all stations of each area in a week range
		sid = area_groups[area]['id'] & em_id & ga_id # get station id in each area
		if len(sid)==0: continue
		em_list = []
		for id_num in sid:
			em_list.append(pd.read_csv(os.path.join(em_path, f'{id_num}_magn.csv')))
		em_data = pd.concat(em_list)
		del em_list
		ga_list = []
		for id_num in sid:
			ga_list.append(pd.read_csv(os.path.join(ga_path, f'{id_num}_sound.csv')))
		ga_data = pd.concat(ga_list)
		del ga_list # release memory 

		## get representative features
		start_stamp = string2stamp(time_range[0])
		em_data = generate_features(em_data, 7, 'magn', start_stamp)
		ga_data = generate_features(ga_data, 7, 'sound', start_stamp)
		
		_final_res = pd.merge(em_data, ga_data, on='Day', how='left')
		_final_res.fillna(0, inplace=True)
		_final_res.drop('Day', axis=1, inplace=True)
		features = _final_res.iloc[-1] # use the last week's features to predict the next week's earthquake
		print(len(features))
		input('check')
		## model prediction
		lgb_model = lgb.Booster(model_file=f'./model/{area}_mag_model.txt')
		predict = np.matrix(lgb_model.predict(features, num_iteration=lgb_model.best_iteration))
		predict = predict[0].argmax(axis=1)
		if predict[0] == 0:
			continue
		else:
			if(predict[0] > max_mag):
				max_mag = predict[0, 0]
				eq_area = area

	magn_level = {0:0, 1:3.7, 2:4.2, 3:4.7, 4:5}
	long = (area_groups[eq_area]['range'][2] + area_groups[eq_area]['range'][3])/2
	lati = (area_groups[eq_area]['range'][0] + area_groups[eq_area]['range'][1])/2
	print(f'magnitude:{magn_level[max_mag]}, longitude:{long}, latitude:{lati}')


