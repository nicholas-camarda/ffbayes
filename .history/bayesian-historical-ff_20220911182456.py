#!python
from enum import unique
import pandas as pd
import numpy as np
import glob
import sys

### Use Monte Carlo simulation to project the score of my team ###
### https://srome.github.io/Making-Fantasy-Football-Projections-Via-A-Monte-Carlo-Simulation/ ###

my_years = [2017, 2018, 2019, 2020, 2021]

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

print("Combined data:", combined_data)
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
        # remove duplicates from historical data, only select name, position, and tm
    return(pd.DataFrame(tm).drop_duplicates(subset=["Name", "Position"])[["Name", "Position", "Tm"]])

tm = make_team(team = my_team, db = combined_data)
print(tm)
print("\n")
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

print(combined_data.columns)
print("\n\n")


def get_games(db, year, week):
    ### return all the players in this week and this year
    result = db[(db["Season"] == year) & (db["G#"] == week)]
    return(result)

def score_player(p, db, year, week):
    sc = db.loc[(db["Name"] == p.Name) & (db["Season"] == year) & (db["G#"] == week)]
    final_sc = sc["FantPt"].reset_index(drop=True)
    return(final_sc)

def get_score_for_player(db, player, years):
    
    # Sample the year and week
    year = np.random.choice(years,
                            # for years 2017-2021
                            p=[0.025, 0.075, 0.15, 0.25, 0.5])
    week = np.random.randint(1,18)

    # Find the player and score them for the given week/year   
    for p in get_games(db, year, week).itertuples():
        if p.Name is None:
            continue
        if player.Name == p.Name:
            sc2 = score_player(p, db, year, week)
            isnan = np.isnan(sc2)
            if (sc2 is None) | is:
                print("Found a nonetype!")
                return 0
            else:
                return(sc2)
    return 0


def simulate(team, db, years, exps=10):
    scores = pd.DataFrame(data=np.zeros((exps,len(team))),
                          columns = [p.Name for p in team.itertuples()])
    print(scores)
    # we are defining our simulator’s projection to be the expected value of the sample mean of the team random variable. 
    for n in range(exps):
        for player in team.itertuples():
            score1 = get_score_for_player(db, player, years)
            scores.at[n, player.Name] += score1
    return scores

outcome = simulate(team = tm, db = combined_data, years = my_years, exps=100)
print(outcome.head())

game_points = outcome.sum(axis=1, skipna=True) # Sum the player scores together

print('Team projection: %s' % game_points.mean())
print('Standard Deviations: %s' % (game_points.std()/np.sqrt(len(outcome.columns))))
