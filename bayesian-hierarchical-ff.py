#!python
import numpy as np
import pandas as pd
import glob
import os
import theano
import pymc3 as pm

### You'll need to enter a conda environemtn to run this.. 
#### conda activate pymc3_env ####
### bayesian hierarchical model to predict fantasy league performance over the next year ###
### https://srome.github.io/Bayesian-Hierarchical-Modeling-Applied-to-Fantasy-Football-Projections-for-Increased-Insight-and-Confidence/ ###

# read in the datasets and combine
all_files = glob.glob(os.path.join("datasets", "*.csv"))
data_temp = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
# make sure it's sorted properly
data = data_temp.sort_values(by = ['Season', 'Name', 'G#'], ascending = [True, True, True]) 

# One-hot-encode the positions
data['pos_id'] = data['Position']
data['position'] = data['Position']
data = pd.get_dummies(data,columns=['position'])

print(data.head)
# Identify teams with integer
ids = np.array([k for k in data['Opp'].unique()])
team_names = ids.copy()
data['opp_team'] = data['Opp'].apply(lambda x : np.where(x == ids)[0][0])
data['team'] = data['Tm'].apply(lambda x : np.where(x == ids)[0][0])

# make a copy of the away column, and call it is_home where 0 == true, and 1 == false
is_home = np.abs(data['Away'] - 1)
data['is_home'] = is_home

pos_ids = np.array([k for k in data['pos_id'].unique()]) # remove the nan
pos_ids_nonan = pos_ids[np.where(pos_ids != "nan")]
onehot_pos_ids = list(map(int, data['pos_id'].isin(pos_ids_nonan)))
data['pos_id'] = onehot_pos_ids

print(data.columns)
# calculate seven game average
# print(help(data.shift))
# print(help(data.expanding))

# So basically you need to do a rolling mean on a groupby. 
# i'm pretty sure these are right, but would be worth confirming with someone more knowledgeable??
num_day_roll_avg = 7
data['7_game_avg'] = data.groupby(['Name', 'Season'])['FantPt'].transform(lambda x: x.rolling(num_day_roll_avg, min_periods=num_day_roll_avg).mean()) 
# print(data['7_game_avg'])

# rank based on the 7-game-avg
ranks = data.groupby(['Name', 'Season'])['7_game_avg'].rolling(num_day_roll_avg, min_periods = num_day_roll_avg).rank(pct = True).mul(4) # mul(4) gives quartile
data['rank'] = ranks.tolist()

data['diff_from_avg'] = data['FantPt'] - data['7_game_avg']

# We are using a single year for the analysis
explore = data[data.apply(lambda x : x['Season'] == ["2017", "2018", "2019"], axis=1)]
train = data[data.apply(lambda x : x['Season'] == 2020, axis=1)]
test = data[data.apply(lambda x : x['Season'] == 2021, axis=1)]

explore.head()


print("Running Model...")

num_positions=4
ranks=4
team_number = len(team_names)
np.random.seed(182)


with pm.Model() as mdl:
    print("Part 1...")
    nu = pm.Exponential('nu minus one', 1/29.,shape=2) + 1 # from https://pymc-devs.github.io/pymc3/notebooks/BEST.html
    err = pm.Uniform('std dev based on rank', 0, 100, shape=ranks)
    err_b = pm.Uniform('std dev based on rank b', 0, 100, shape=ranks)

    print("Part 2...")
    # Theano shared variables to change at test time
    player_home = theano.shared(np.asarray(train['is_home'].values, dtype = int))
    player_avg = theano.shared(np.asarray((train['7_game_avg']).values, dtype = float))
    player_opp = theano.shared(np.asarray((train['opp_team']).values, dtype = int))
    player_team = theano.shared(np.asarray((train['team']).values, dtype = int))
    player_rank = theano.shared(np.asarray((train['rank']-1).values, dtype = int))
    qb = theano.shared(np.asarray((train['position_QB']).values.astype(int), dtype = int))
    wr = theano.shared(np.asarray((train['position_WR']).values.astype(int), dtype = int))
    rb = theano.shared(np.asarray((train['position_RB']).values.astype(int), dtype = int))
    te = theano.shared(np.asarray((train['position_TE']).values.astype(int), dtype = int))
    pos_id = theano.shared(np.asarray((train['pos_id']).values, dtype = int))

    print("Part 3...")
    # Defensive ability of the opposing team vs. each position, partially pooled
    opp_def = pm.Normal('opp team prior',0, sd=100**2, shape=num_positions)
    opp_qb = pm.Normal('defensive differential qb', opp_def[0], sd=100**2, shape=team_number)
    opp_wr = pm.Normal('defensive differential wr', opp_def[1], sd=100**2, shape=team_number)
    opp_rb = pm.Normal('defensive differential rb', opp_def[2], sd=100**2, shape=team_number)
    opp_te = pm.Normal('defensive differential te', opp_def[3], sd=100**2, shape=team_number)
    
    print("Part 4...")
    # Partially pooled ability of the player's rank partially pooled based on position
    home_adv = pm.Normal('home additivie prior', 0, 100**2,shape = num_positions)     
    away_adv = pm.Normal('away additivie prior', 0, 100**2,shape = num_positions)     
    pos_home_qb = pm.Normal('home differential qb',home_adv[0],10**2, shape = ranks)
    pos_home_rb = pm.Normal('home differential rb',home_adv[1],10**2, shape = ranks)
    pos_home_te = pm.Normal('home differential te',home_adv[2],10**2, shape = ranks)
    pos_home_wr = pm.Normal('home differential wr',home_adv[3],10**2, shape = ranks)
    pos_away_qb = pm.Normal('away differential qb',away_adv[0],10**2, shape = ranks)
    pos_away_rb = pm.Normal('away differential rb',away_adv[1],10**2, shape = ranks)
    pos_away_wr = pm.Normal('away differential wr',away_adv[2],10**2, shape = ranks)
    pos_away_te = pm.Normal('away differential te',away_adv[3],10**2, shape = ranks)

    print("Part 5...")
    # First likelihood where the player's difference from average is explained by defensive abililty
    def_effect = qb*opp_qb[player_opp]+ wr*opp_wr[player_opp]+ rb*opp_rb[player_opp]+ te*opp_te[player_opp]
    like1 = pm.StudentT('Diff From Avg', mu=def_effect, sd=err_b[player_rank],nu=nu[1], observed = train['diff_from_avg'])
    
    print("Part 6...")
    # Second likelihood where the score is predicted by defensive power plus other smaller factors
    mu = player_avg + def_effect
    mu += rb*pos_home_rb[player_rank]*(player_home) + wr*pos_home_wr[player_rank]*(player_home) 
    mu += qb*pos_home_qb[player_rank]*(player_home) + te*pos_home_te[player_rank]*(player_home) 
    mu += rb*pos_away_rb[player_rank]*(1-player_home) + wr*pos_away_wr[player_rank]*(1-player_home) 
    mu += qb*pos_away_qb[player_rank]*(1-player_home) + te*pos_away_te[player_rank]*(1-player_home) 
    like2 = pm.StudentT('Score', mu=mu, sd=err[player_rank], nu=nu[0], observed=train['score'])

    print("Part 7: Training...")
    # Training!
    trace=pm.sample(10000, pm.Metropolis())
    
tr=trace[-5000::3]
pm.traceplot(tr)
