import theano
import pymc3 as pm
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import numpy as np

# what does ppc look like?
with open(r"model.pickle", "rb") as input_file:
    mdl = pickle.load(input_file)
    tr = mdl["trace"]
    
    ppc = mdl["ppc"]
    print(ppc['FantPt'].mean(axis = 0))
    print(len(ppc['FantPt'].mean(axis = 0)))
    print(ppc['FantPt'].std(axis = 0))
    print(ppc['Diff From Avg'].mean(axis = 0))
    print(ppc['Diff From Avg'].std(axis = 0))

    test = mdl["test"]
    print(test)
    print(len(test))
    
    proj_mae = mean_absolute_error(y_true = test.loc[:, "7_game_avg"], y_pred = ppc["FantPt"].mean(axis=0), multioutput='raw_values')
    h_proj_mae = mean_absolute_error(y_true = test.loc[:, "FantPt"], y_pred = test.loc[:, "7_game_avg"], multioutput='raw_values')
    proj_sd = ppc["FantPt"].std(axis = 0)
    d = pd.DataFrame({"proj MAE": proj_mae,
                      "historical avg MAE": ,
                      "sd": })
    
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