## Environment Requirement
library & version
* lightgbm                  2.3.0          
* numpy                     1.19.3          
* requests                  2.22.0           
* requests-oauthlib         1.3.0  
* pandas                    1.1.3

## Dataset Preparation
unzip the trainning data and put all csv files into the `data` folder;
Move the eqlst.csv and StationInfo.csv to data folder.

## Training&Inference
1. Run `mergeData.py` to generate pkl file under ‘data’ folder.
2. Run `readData.py` to caculate features. The generated features file will be saved in 'area_feature' folder.
3. Run `lgb.py` to generate models for different region.
4. Replace the token in pred.py by your team's token. Then run `pred.py` to get the prediction results.

## Note
* This baseline will give a basic model. In addtition, the way of how to get data and update result by token is also given. 
* We divide the target area into 8 small areas, the stations in one area will be considered as a group. 
A time window will slid on average_sound and average_magn feature of this group to get some statistical characteristics, such as the max, min,and mean of a week (day granularity).
* For the label, we choose earthquakes of the next week (from next monday to next sunday). However, the Sunday won't be included because we can't get its data when we do prediciton. The prediction should be updated on Sunday (Chinese standard time: UTC+8). 
* Area with the max magnitude will be used as the final result, and its region center will be the predicted epicenter.