#!python
from enum import unique
import pandas as pd
import numpy as np
import glob
from functools import lru_cache
import nflgame

### Use Monte Carlo simulation to project the score of my team ###
### https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/ ###

my_team = pd.read_csv("my_team_2022.tsv", sep='\t')["Name"]
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


scoring = {
    'passing_yds' : lambda x : x*.04 +
                        (3. if x >= 300 else 0),
    'passing_tds' : lambda x : x*4., 
    'passing_ints' : lambda x : -1.*x,
    'rushing_yds' : lambda x : x*.1 + (3 if x >= 100 else 0),
    'rushing_tds' : lambda x : x*6.,
    'kickret_tds' : lambda x : x*6.,
    'receiving_tds' : lambda x : x*6.,
    'receiving_yds' : lambda x : x*.1,
    'receiving_rec' : lambda x : x,
    'fumbles_lost' : lambda x : -1*x,
    'passing_twoptm'  : lambda x : 2*x,
    'rushing_twoptm' : lambda x : 2*x,
    'receiving_twoptm' : lambda x : 2*x
}

def score_player(player):
    score = 0
    for stat in player._stats:
        if stat in scoring:
            score += scoring[stat](getattr(player,stat))    
    return score


def get_score_for_player(player):
    
    # Sample the year and week
    year = np.random.choice([2017, 2018, 2019, 2020, 2021],
                            p=[.025, 0.075, 0.15, 0.25, 0.5])
    week = np.random.randint(1,18)
    
    # Find the player and score them for the given week/year 
    games = get_games(year,week)
    for p in games:
        if p.player is None:
            continue
        if player == p.player:
            return score_player(p)
        
    return get_score_for_player(player) # Retry due to bye weeks / failure for any other reason


@lru_cache(200) # Define a cache with 200 empty slots
def get_games(year,week):
    g = nflgame.games(year=year,week=week)
    return nflgame.combine_game_stats(g)


def simulate(team, exps=10):
    scores = pd.DataFrame(data=np.zeros((exps,len(team))),
                          columns = [p.name for p in team])
    for n in range(exps):
        for player in team:
            scores.loc[n, player.name] += get_score_for_player(player)
    return scores

g = get_games(year = 2017, week = 1)
print(g)
print([p.player for p in g])

outcome = simulate(tm, exps=100)
outcome.head()


game_points = outcome.sum(axis=1, skipna=True) # Sum the player scores together

print('Team projection: %s' % game_points.mean())
print('Standard Deviations: %s' % (game_points.std()/np.sqrt(len(outcome.columns))))
