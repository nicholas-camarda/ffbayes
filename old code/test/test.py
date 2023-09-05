import nflgame
import os

os.mkdir("datasets")
yr = 2018

df = nflgame.combine(nflgame.games(yr))
df.csv('datasets/season{}.csv'.format(yr))


