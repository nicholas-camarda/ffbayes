#!python
import numpy as np
import pandas as pd
import glob
import os
import theano
import pymc3 as pm
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import multiprocessing

cores = 7 # multiprocessing.cpu_count() - 1
print("Using %d cores" % cores)

### You'll need to enter a conda environemtn to run this.. 
#### conda activate pymc3_env ####
### bayesian hierarchical model to predict fantasy league performance over the next year ###
### https://srome.github.io/Bayesian-Hierarchical-Modeling-Applied-to-Fantasy-Football-Projections-for-Increased-Insight-and-Confidence/ ###

def pickle_model(output_path: str, model, trace, ppc):
        """Pickles PyMC3 model and trace"""
        with open(output_path, "wb") as buff:
            pickle.dump({"model": model, "trace": trace, "ppc": ppc}, buff)
        
def bayesian_hierarchical_ff(cores):
    # read in the datasets and combine
    all_files = glob.glob(os.path.join("datasets", "*.csv"))
    data_temp = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    # make sure it's sorted properly
    data = data_temp.sort_values(by = ['Season', 'Name', 'G#'], ascending = [True, True, True]) 

    # One-hot-encode the positions
    data['pos_id'] = data['Position']
    data['position'] = data['Position']
    data = pd.get_dummies(data,columns=['position'])

    # print(data.head)
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

    # print(data.columns)
    # calculate seven game average
    # print(help(data.shift))
    # print(help(data.expanding))

    # So basically you need to do a rolling mean on a groupby. 
    # i'm pretty sure these are right, but would be worth confirming with someone more knowledgeable??
    num_day_roll_avg = 7
    data['7_game_avg'] = data.groupby(['Name', 'Season'])['FantPt'].transform(lambda x: x.rolling(num_day_roll_avg, min_periods=num_day_roll_avg).mean()) 
    # print(data['7_game_avg'])

    # rank based on the 7-game-avg
    # rolling(num_day_roll_avg, min_periods = num_day_roll_avg)
    ranks = data.groupby(['Name', 'Season'])['7_game_avg'].rank(pct = False, method = 'average')# mul(4) gives quartile
    quartile_ranks = pd.qcut(ranks, 4, labels = False, duplicates = 'drop')
    data['rank'] = quartile_ranks.tolist()
    # print(data['rank'].head)

    data['diff_from_avg'] = data['FantPt'] - data['7_game_avg']

    # removea all NA
    data = data.dropna(axis = 0)
    # convert rank column to integer
    data = data.astype({'rank':int})

    # We are using a single year for the analysis
    explore = data[data.apply(lambda x : x['Season'] == ["2017", "2018", "2019"], axis=1)]
    train = data[data.apply(lambda x : x['Season'] == 2020, axis=1)]
    test = data[data.apply(lambda x : x['Season'] == 2021, axis=1)]

    pickle_path = "model/ff_2021.pickle"
    try:
        mdl_obj = pickle.load(open(pickle_path, "rb"))
        mdl = mdl_obj.model
        
    except (OSError, IOError) as e:
        pickle_model(output_path = pickle_path, model = mdl, trace = tr, ppc = ppc)
        
    print("Running Model...")

    num_positions=4
    ranks=4
    team_number = len(team_names)
    np.random.seed(182)

    with pm.Model() as mdl:
        print("Part 1... Define observables")
        nu = pm.Exponential('nu minus one', 1/29.,shape=2) + 1 # from https://pymc-devs.github.io/pymc3/notebooks/BEST.html
        err = pm.Uniform('std dev based on rank', 0, 100, shape=ranks)
        err_b = pm.Uniform('std dev based on rank b', 0, 100, shape=ranks)

        print("Part 2... Theano shared variables to change at test time")
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

        print("Part 3... Defensive ability of the opposing team vs. each position, partially pooled")
        # Defensive ability of the opposing team vs. each position, partially pooled
        opp_def = pm.Normal('opp team prior',0, sd=100**2, shape=num_positions)
        opp_qb = pm.Normal('defensive differential qb', opp_def[0], sd=100**2, shape=team_number)
        opp_wr = pm.Normal('defensive differential wr', opp_def[1], sd=100**2, shape=team_number)
        opp_rb = pm.Normal('defensive differential rb', opp_def[2], sd=100**2, shape=team_number)
        opp_te = pm.Normal('defensive differential te', opp_def[3], sd=100**2, shape=team_number)
        
        print("Part 4... Partially pooled ability of the player's rank partially pooled based on position")
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

        print("Part 5...  First likelihood where the player's difference from average is explained by defensive abililty")
        # First likelihood where the player's difference from average is explained by defensive abililty
        def_effect = qb*opp_qb[player_opp]+ wr*opp_wr[player_opp]+ rb*opp_rb[player_opp]+ te*opp_te[player_opp]
        like1 = pm.StudentT('Diff From Avg', mu=def_effect, sd=err_b[player_rank], nu=nu[1], observed = train['diff_from_avg'])
        
        print("Part 6... Second likelihood where the score is predicted by defensive power plus other smaller factors")
        # Second likelihood where the score is predicted by defensive power plus other smaller factors
        mu = player_avg + def_effect
        mu += rb*pos_home_rb[player_rank]*(player_home) + wr*pos_home_wr[player_rank]*(player_home) 
        mu += qb*pos_home_qb[player_rank]*(player_home) + te*pos_home_te[player_rank]*(player_home) 
        mu += rb*pos_away_rb[player_rank]*(1-player_home) + wr*pos_away_wr[player_rank]*(1-player_home) 
        mu += qb*pos_away_qb[player_rank]*(1-player_home) + te*pos_away_te[player_rank]*(1-player_home) 
        like2 = pm.StudentT('FantPt', mu=mu, sd=err[player_rank], nu=nu[0], observed=train['FantPt'])

        print("Part 7... Training!")
        # Training!
        trace=pm.sample(draws=10000, step=pm.Metropolis(), 
                        progressbar=True, 
                        return_inferencedata=False, # return a MultiTrace obj
                        cores=cores)
        
        tr=trace[-5000::3]
        
        # save the tracings into 'plots' directory
        os.makedirs("plots", exist_ok=True)
        trarr = pm.plot_trace(tr)
        fig = plt.gcf() # to get the current figure...
        fig.savefig("plots/training_traces.png", dpi = 300) # and save it directly
        
        # TODO: mabye define a class to save these results into??
        
        # evaluate the model
        print("Evaluating the model...")
        player_home.set_value(np.asarray(test['is_home'].values, dtype = int))
        player_avg.set_value(np.asarray((test['7_game_avg']).values, dtype = float))
        player_opp.set_value(np.asarray((test['opp_team']).values, dtype = int))
        player_rank.set_value(np.asarray((test['rank']-1).values, dtype = int))
        pos_id.set_value(np.asarray((test['pos_id']).values, dtype = int))
        player_team.set_value(np.asarray((test['team']).values, dtype = int))
        qb.set_value(np.asarray((test['position_QB']).values.astype(int), dtype = int))
        wr.set_value(np.asarray((test['position_WR']).values.astype(int), dtype = int))
        rb.set_value(np.asarray((test['position_RB']).values.astype(int), dtype = int))
        te.set_value(np.asarray((test['position_TE']).values.astype(int), dtype = int))

        print("Sampling from the posterior...")
        ppc=pm.sample_posterior_predictive(tr, samples=1000, model= mdl)
        
        print('Projection Mean Absolute Error:', mean_absolute_error(test.loc[:,'FantPt'].values, ppc['FantPt'].mean(axis=0)))
        print('7 Day Average Mean Absolute Error:', mean_absolute_error(test.loc[:,'FantPt'].values, test.loc[:,'7_game_avg'].values))

        # i think i need to calculate on ppc, the MAE of each sample and the sd of each sample
        # i need to do something similar for the historical average, but not sure what counts as a sample
        # DEBUG: what is 'd' ?????
        d = pd.DataFrame({'proj MAE':  mean_absolute_error(test.loc[:,'FantPt'].values, ppc['FantPt'], multioutput='raw_values'), 
                        'historical avg MAE': mean_absolute_error(test.loc[:,'FantPt'].values, test.loc[:,'7_game_avg'].values, multioutput='raw_values'),
                        'sd' : err[player_rank]})
        max_sd = d['sd'].max()
        plt.figure(figsize=(8,5))
        ax=plt.gca()
        ax.plot(np.linspace(0,max_sd,30), np.array([d[d['sd'] <= k]['proj MAE'].mean() for k in np.linspace(0,max_sd,30)]))
        ax.plot(np.linspace(0,max_sd,30), np.array([d[d['sd'] <= k]['historical avg MAE'].mean() for k in np.linspace(0,max_sd,30)]), color='r')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xlabel('Standard Deviation Cutoff')
        ax.set_title('MAE for Projections w/ SDs Under Cutoff')
        ax.legend(['Bayesian Mean Projection', 'Rolling 7 Game Mean'], loc=4)
        fig = plt.gcf() # to get the current figure...
        fig.savefig("plots/MAE_for_projections_w_sd_under_cutoff.png", dpi = 300) # and save it directly
        
        print("Drawing conclusions from the model...")
        t = pd.DataFrame({'projection':  tr['defensive differential rb'].mean(axis=0), 'sd' : tr['defensive differential rb'].std(axis=0),'name': team_names})
        f=plt.figure(figsize=(8,10))
        plt.errorbar(x=t['projection'],y=range(1,len(t)+1),xerr=t['sd'], lw=3, fmt='|')
        plt.title('Team Effect\'s on RB Point Average (2021)')
        end=plt.yticks(range(1,len(t)+1), [name for name in t['name']])
        plt.xlim([-6,8])
        plt.xlabel('Change in opponent\'s RB\'s average')
        fig = plt.gcf() # to get the current figure...
        fig.savefig("plots/RB.png", dpi = 300) # and save it directly
        
        t = pd.DataFrame({'projection':  tr['defensive differential qb'].mean(axis=0), 'sd' : tr['defensive differential qb'].std(axis=0),'name': team_names})
        f=plt.figure(figsize=(8,10))
        plt.errorbar(x=t['projection'],y=range(1,len(t)+1),xerr=t['sd'], lw=3, fmt='|')
        plt.title('Team\'s Effect on QB Point Average (2021)')
        end=plt.yticks(range(1,len(t)+1), [name for name in t['name']])
        plt.xlim([-11.5,10])
        plt.xlabel('Change in opponent\'s QB\'s average')
        fig = plt.gcf() # to get the current figure...
        fig.savefig("plots/QB.png", dpi = 300) # and save it directly
        
        t = pd.DataFrame({'projection':  tr['defensive differential te'].mean(axis=0), 'sd' : tr['defensive differential te'].std(axis=0),'name': team_names})
        f=plt.figure(figsize=(8,10))
        plt.errorbar(x=t['projection'],y=range(1,len(t)+1),xerr=t['sd'], lw=3, fmt='|')
        plt.title('Team Effect\'s on TE Point Average (2021)')
        end=plt.yticks(range(1,len(t)+1), [name for name in t['name']])
        plt.xlim([-8,8])
        plt.xlabel('Change in opponent\'s TE\'s average')
        fig = plt.gcf() # to get the current figure...
        fig.savefig("plots/TE.png", dpi = 300) # and save it directly
        
        t = pd.DataFrame({'projection':  tr['defensive differential wr'].mean(axis=0), 'sd' : tr['defensive differential wr'].std(axis=0),'name': team_names})
        f=plt.figure(figsize=(8,10))
        plt.errorbar(x=t['projection'],y=range(1,len(t)+1),xerr=t['sd'], lw=3, fmt='|')
        plt.title('Team\'s Effect on WR Point Average (2021)')
        end=plt.yticks(range(1,len(t)+1), [name for name in t['name']])
        plt.xlim([-4.5,3.1])
        plt.xlabel('Change in opponent\'s WR\'s average')
        fig = plt.gcf() # to get the current figure...
        fig.savefig("plots/WR.png", dpi = 300) # and save it directly
    print("Done!")
    return(tr)

if __name__ == "__main__":
    bayesian_hierarchical_ff(cores)