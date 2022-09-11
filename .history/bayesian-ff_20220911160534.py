#!python
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


def make_team(team, db):
    ### Make my team based off of my picks and whether they have historical data to simulate on ###
    tm = []
    for plr in db:
        print(plr)
        if plr["Name"] not in team["Name"] or plr["Position"] not in set(['QB','WR','TE','RB']):
            continue
        tm.append(plr)
    return(pd.DataFrame(tm))

def validate_team(team, my_team):
    ### Check which members of my team actually have historical data to simulate on ###
    # get the column names of team using team.dtype.names
    pd_frame = pd.DataFrame(team)
    unique_teams = pd_frame.loc[:, ["Name","Position","Tm"]].drop_duplicates()
    r_unique_teams = unique_teams.to_records()
    for t in r_unique_teams:
        print(t.Name, t.Position, t.Tm)
    
    db_set = set(r_unique_teams.Name)
    my_team_set = set(my_team.Name)
    if db_set == my_team_set:
        print("Found all team members.")
    else:
        print("\nMissing team members.")
        print(my_team_set.difference(db_set))
        

tm = make_team(team = my_team, db = combined_data)
validate_team(team = tm, my_team = my_team)

# convert to pandas df
pd_tm = pd.DataFrame(tm)

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
    return(db[db.year == year and db.week == week])
        

g = get_games(db = combined_data, year = 2018, week = 1)
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
