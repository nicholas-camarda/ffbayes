
from sklearn.metrics import mean_absolute_error
import pandas as pd
test = pd.DataFrame({'FantPt':[[0.5, 1], [-1, 1], [7, -6]]})
ppc = pd.DataFrame({'FantPt': [[0, 2], [-1, 2], [8, -5]]

mae = mean_absolute_error(test, ppc, multioutput='raw_values')

print(mae)