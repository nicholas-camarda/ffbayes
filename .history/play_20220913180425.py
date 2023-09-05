import theano
import pymc3 as pm
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle

# what does ppc look like?
mdl = pickle.load("model.pickle", "rb")
print(mdl.ppc)