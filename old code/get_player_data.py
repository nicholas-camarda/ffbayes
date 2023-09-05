#!python
import nflgame
import pandas as pd

my_team = pd.read_csv("my_team_2022.tsv", sep = '\t')

def make_team(team):
    ### grab nfl game data from your team ###
    tm = pd.DataFrame(columns = ["full_name", "team"])
    for p in team:
        for plr in nflgame.find(p):
            # if plr.position not in set(['QB','WR','TE','RB']) or plr.status == '':
            #     continue
            if isinstance(plr, str):
                print("string")
                splt_plr = plr.split(" ")
                new_player = pd.DataFrame({"full_name":[splt_plr[0]], "team":[splt_plr[1]]})
                tm = pd.concat([tm, new_player])
            else:
                print("object")
                new_player = pd.DataFrame({"full_name":[plr.name], "team":[plr.position]})
                tm = pd.concat([tm, new_player])
    return tm

def validate_team(team):
    for t in team:
        print(t.full_name, t.team)
        
tm = make_team(my_team.Name)
print(tm)
# validate_team(tm)

print(nflgame.find("Jalen Hurts"))

