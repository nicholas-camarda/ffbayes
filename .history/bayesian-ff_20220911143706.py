#!python
import pandas as pd
import numpy as np
import glob

### Use Monte Carlo simulation to project the score of my team ###

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


def score_player(player):
    score = 0
    for stat in player._stats:
        if stat in scoring:
            score += scoring[stat](getattr(player,stat))    
    return score


def get_combined_data(directory_path):
    # Get data file names
    files = glob.glob(directory_path + "/*.csv")
    dfs = list()
    for f in files:
        data = pd.read_csv(f)
        dfs.append(data)

    df = pd.concat(dfs, axis = 0, ignore_index = True)
    # convert each row of df into objects
    recarray = df.to_records()
    return(recarray)

combined_data = get_combined_data(directory_path = "datasets")
# convert my team csv into objects also
my_team = pd.read_csv("my_team_2022.tsv", sep='\t').to_records()

# Validate team, filter combined object down to team and opponent team
# print(combined_data)


def make_team(team, db):
    tm = []
    for plr in db:
        if plr.Name not in team.Name or plr.Position not in set(['QB','WR','TE','RB']):
            continue
        tm.append(plr)
    return(np.unique(tm))

def validate_team(team, my_team):
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
# access the columns of our np.recarray

validate_team(team = tm, my_team = my_team)


# project both, optimize
