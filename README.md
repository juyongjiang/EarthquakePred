# A Simple but Effective MLP Model for AETA Earthquake Prediction
Following privious work to mitigate the error accumulation between differnt stations, I roughly divide target region into 8 small areas (defined in `area_map.py`). 
Thus, the stations in one area will be considered as a group.
My method is built based on the useful resources provided by privious competitor.

## Installation
Clone this project:

```bash
git clone git@github.com:juyongjiang/Earthquake_Pred.git
```

Make sure you have `Python>=3.8` and `Pytorch>=1.8` installed on your machine. 

* Pytorch 1.8.1
* Python 3.8.*

Install python dependencies by running:

```bash
conda env create -f requirements.yml
# After creating environment, activate it
conda activate automl
```

## Datasets Preparation
Download all data provided by [competition platform](https://tianchi.aliyun.com/competition/entrance/531972/information). Then, 
* Unzip `EM_TRAIN_1.zip`, `EM_TRAIN_2.zip`, `EM_TRAIN_3.zip` and `GA_TRAIN_1.zip`, `GA_TRAIN_2.zip`, `GA_TRAIN_3.zip`, then move all csv files of `EM_*/*.csv` and `GA_*/*.csv` to `dataset/raw/EM_GA_DATA`; [The corresponding earthquake data (view as label) `EC_TRAINSET.csv` move to `dataset/raw/`]
* Unzip `EM_TEST.zip` and `GA_TEST.zip`, move them to `dataset/`; [The corresponding earthquake data `EC_TESTSET.csv` move to `dataset/`]
* Unzip `README.zip` to get `StationInfo.csv` and `FeatureInfo.xlsx`, then move them to `dataset/`;
* Unzip `AETA_20220529-20220702.zip`, then move to `dataset/`.

Then, run the following command to preprocess TRAIN data to get the representitive **features** and **labels**:
```bash
python preprocess.py
```
Specifically, a time window (7 days) will slide on average_sound (sound@abs_mean)  and average_magn (magn@abs_mean) feature of each group to get some statistical characteristics, such as the max, min, and mean of a week (day granularity).
```python
'average_day_max_mean_magn', 'average_day_min_mean_magn', 'average_day_mean_max_magn', 'average_day_mean_min_magn', 
'average_mean_magn', 'average_max_magn', 'average_min_magn', 'average_max_min_magn', 
'average_lastday_mean_magn', 'average_lastday_max_magn', 'average_lastday_min_magn', 'average_lastday_max_min_magn', 
'diff_1_day_max_mean_magn', 'diff_1_day_min_mean_magn', 'diff_1_day_mean_max_magn', 'diff_1_day_mean_min_magn', 
'diff_1_mean_magn', 'diff_1_max_magn', 'diff_1_min_magn', 'diff_1_max_min_magn', 
'diff_1_lastday_mean_magn', 'diff_1_lastday_max_magn', 'diff_1_lastday_min_magn', 'diff_1_lastday_max_min_magn'
```

For the label setting, I choose earthquakes of the next week (during the next monday to next sunday). However, the Sunday won't be included because I can't get its data when making prediciton. The prediction will be updated on Sunday (Chinese standard time: UTC+8). Furthermore, regression predition may conduce significant errors. So I discrete the continuous Magnitude data into integer class data according to the range it belongs to.
```python
magnitude < 3.5 -> label = 0
magnitude < 4.0 -> label = 1
magnitude < 4.5 -> label = 2
magnitude < 5.0 -> label = 3
magnitude >= 5.0 -> label = 4
```

Finally, the path structure of dataset folder will be like:
```bash
$ tree
.
├── dataset
    ├── AETA_20220529-20220702
    ├── processed
        ├──AREA_FEATURE
        ├──magn_data.pkl
        └──sound_data.pkl
    ├── raw
        ├──EM_GA_DATA
        └──EC_TRAINSET.csv
    ├──EM_TEST
    ├──GA_TEST
    ├──EC_TESTSET.csv
    ├──FeatureInfo.xlsx
    ├──StationInfo.csv
    └──README.md                // introduce the details of station information and data characteristics
...
``` 

## Training and Predition 
Training a simple but effective MLP model (defined in `model/mlp.py`) on TRAIN data. Since the area feature is contructed by each area, it will have an individual model for each area, named as `saved/eqmodel-area.pth`.
Please run the following command:

```bash
python train.py
```

In predition stage, the area with the max magnitude will be used as the final prediction result, and the center of the area it belongs to will be the predicted epicenter of earthquake. The submission file `prediction.csv` will be generated automatically. To check the difference between prediction and ground truth, a `ground_truth.csv` file will also be generated.  
Please run the following command:

```bash
python pred.py
```

## LICENSE
```bash
MIT License

Copyright (c) 2022 Juyong JIANG

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```