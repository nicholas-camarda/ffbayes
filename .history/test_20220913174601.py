
from sklearn.metrics import mean_absolute_error

mean_absolute_error(test.loc[:,'FantPt'].values, ppc['FantPt'], multioutput='raw_values')