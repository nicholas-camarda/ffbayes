import theano
import pymc3 as pm
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle

# what does ppc look like?
with open(r"model.pickle", "rb") as input_file:
    mdl = pickle.load(input_file)
    tr = mdl["trace"]
    print(tr['sd']
    
    ppc = mdl["ppc"]
    print(ppc['FantPt'].mean(axis = 0))
    print(ppc['FantPt'].std(axis = 0))
    print(ppc['Diff From Avg'].mean(axis = 0))
    print(ppc['Diff From Avg'].std(axis = 0))
    
    test = mdl["test"]
    print(test)