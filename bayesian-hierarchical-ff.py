#!python

import numpy as np
import pandas as pd
import glob
import os

### bayesian hierarchical model to predict fantasy league performance over the next year ###
### https://srome.github.io/Bayesian-Hierarchical-Modeling-Applied-to-Fantasy-Football-Projections-for-Increased-Insight-and-Confidence/ ###

# read in the datasets and combine
all_files = glob.glob(os.path.join("datasets", "*.csv"))
data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True) # why parenthesis instead of list?
print(data.head)

# One-hot-encode the positions
data['pos_id'] = data['Position']
data = pd.get_dummies(data,columns=['Position'])

# Identify teams with integer
ids = np.array([k for k in data['Opp'].unique()])
team_names = ids.copy()
data['Opp'] = data['Opp'].apply(lambda x : np.where(x == ids)[0][0])
data['Tm'] = data['Tm'].apply(lambda x : np.where(x == ids)[0][0])

pos_ids = np.array([k for k in data['pos_id'].unique()]) # remove the nan
pos_ids_nonan = pos_ids[np.where(pos_ids != "nan")]
onehot_pos_ids = list(map(int, data['pos_id'].isin(pos_ids_nonan)))
data['pos_id'] = onehot_pos_ids

# calculate sever game average

data['diff_from_avg'] = data['FantPt'] - data['7_game_avg']

# We are using a single year for the analysis
explore = data[data.apply(lambda x : x['year'].isin(["2017", "2018", "2019"]), axis=1)]
train = data[data.apply(lambda x : x['year'] == 2020, axis=1)]
test = data[data.apply(lambda x : x['year'] == 2021, axis=1)]

explore.head()

