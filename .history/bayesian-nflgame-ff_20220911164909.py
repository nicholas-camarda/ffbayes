#!python
from enum import unique
import pandas as pd
import numpy as np
import glob
from functools import lru_cache
import nflgame

### Use Monte Carlo simulation to project the score of my team ###
### https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/ ###

my_team = pd.read_csv("my_team_2022.tsv", sep='\t')
# print(my_team)

def make_team(team):
    tm = []
    for p in team:
        for plr in nflgame.find(p):
            if plr.position not in set(['QB','WR','TE','RB']) or plr.status == '':
                continue
            tm.append(plr)
    return(tm)

def validate_team(team):
    for t in team:
        print(t.full_name, t.team)

tm = make_team(my_team)
validate_team(tm)

