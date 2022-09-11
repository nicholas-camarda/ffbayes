#!python
from enum import unique
import pandas as pd
import numpy as np
import glob
from functools import lru_cache
import nflgame

### Use Monte Carlo simulation to project the score of my team ###
### https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/ ###

def get_combined_data(directory_path):
    # Get data file names
    files = glob.glob(directory_path + "/*.csv")
    dfs = list()
    for f in files:
        data = pd.read_csv(f)
        dfs.append(data)

    df = pd.concat(dfs, axis = 0, ignore_index = True)
    # return a pandas dataframe
    return(df)

combined_data = get_combined_data(directory_path = "datasets")
my_team = pd.read_csv("my_team_2022.tsv", sep='\t')

# print("Combined data:", combined_data)
# print("Team:", my_team)

def make_team(team, db):
    ### Make my team based off of my picks and whether they have historical data to simulate on ###
    tm = []
    my_team_names = set(team.Name)
    valid_positions = set(['QB','WR','TE','RB'])
    for plr in db.itertuples(index=True, name='Pandas'):
        if plr.Name not in my_team_names or plr.Position not in valid_positions:
            continue
        tm.append(plr)
    return(pd.DataFrame(tm))

tm = make_team(team = my_team, db = combined_data)

def validate_team(db_team, my_team):
    ### Check which members of my team actually have historical data to simulate on ###
    # get the column names of team using team.dtype.names
    unique_teams = db_team.loc[:, ["Name","Position","Tm"]].drop_duplicates()
    
    db_set = set(unique_teams.Name)
    my_team_set = set(my_team.Name)
    print("Players to project: ")
    print(unique_teams.Name)
    
    if db_set == my_team_set:
        print("Found all team members.")
    else:
        print("\nMissing team members:")
        print(my_team_set.difference(db_set))

validate_team(db_team = tm, my_team = my_team)


# scoring dictionary
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

@lru_cache(200) # Define a cache with 200 empty slots
def get_games(db, year, week):
    ### return all the players in this week and this year
    result = db[db["year"] == year and db["week"] == week]
    return(result)
        

g = get_games(db = combined_data, year = 2018, week = 0)
print(g)




def score_player(player):
    score = 0
    for stat in player._stats:
        if stat in scoring:
            score += scoring[stat](getattr(player,stat))    
    return score

def get_score_for_player(player, years):
    
    # Sample the year and week
    year = np.random.choice(years,
                            # for years 2017-2021
                            p=[0.05, 0.1, 0.15, 0.3, 0.5])
    week = np.random.randint(0,17)
    
    # Find the player and score them for the given week/year   
    for p in get_games(year,week):
        if player == p.player:
            return score_player(p)
        
    return get_score_for_player(player) # Retry due to bye weeks / failure for any other reason


# project both, optimize
