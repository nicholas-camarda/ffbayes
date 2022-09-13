import theano
import pymc3 as pm
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle

# what does ppc look like?
with open(r"model.pickle", "rb") as input_file:
    mdl = pickle.load(input_file)
    ppc = mdl["ppc"])