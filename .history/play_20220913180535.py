import theano
import pymc3 as pm
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle

# what does ppc look like?with open(r"someobject.pickle", "rb") as input_file:
   ...:     e = cPickle.load(input_file)
mdl = pickle.load("model.pickle")
print(mdl.ppc)