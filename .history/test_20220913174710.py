
from sklearn.metrics import mean_absolute_error
test = [[0.5, 1], [-1, 1], [7, -6]]
ppc = [[0, 2], [-1, 2], [8, -5]]

mean_absolute_error(test, ppc, multioutput='raw_values')