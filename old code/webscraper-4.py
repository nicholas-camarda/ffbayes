import nflgame
import pandas as pd
import numpy as np
from functools import lru_cache
import os
from pathlib import Path
from datetime import datetime
from sys import argv
from requests import get #  HTTP requests straight from Python.
from bs4 import BeautifulSoup # parse the raw html

# this version works! 08/11/20

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
    for stat in player.stats:
        if stat in scoring:
            score += scoring[stat](getattr(player,stat))    
    return score

# get avg fantasy score for previous k games for player given
# starting from a given week and year
# return avg fantasy league performance for past k games
def get_past_player_fantasy_avg(pname, wk, yr, k=7):
    # wk = 1; yr = 2018
    if wk < 8:
        yr = yr - 1
        wk1 = 18 + wk - 7
        lm = list(range(wk1, 18)) + list(range(1, wk+1))
        # print(yr, wk1, lm)
    else:
        lm = list(range(wk-7, wk))
        # print(yr, wk, lm)


    p_all_wk_score = []
    for w in lm:
        g = nflgame.games(yr, week=w)
        ps = nflgame.combine_game_stats(g)
        
        
        for p in ps:
            # print("name ",p.name)
            if p.player is None:
                continue            
            else:
                if p.name == pname:
                    p_all_wk_score.append(score_player(p))
    m = np.mean(p_all_wk_score)
    return(m)


# get fantasy rank for previous k games for player given
# starting from a given week and year
# return rank fantasy league for past k games
def get_past_player_fantasy_rank(pname, wk, yr, k=7):
    # wk = 1; yr = 2018
    if wk < 8:
        yr = yr - 1
        wk1 = 18 + wk - 7
        lm = list(range(wk1, 18)) + list(range(1, wk+1))
        print(yr, wk1, lm)
    else:
        lm = list(range(wk-7, wk))
        print(yr, wk, lm)
        
    everyone_score = []
    p_all_wk_idx = []
    for w in lm:
        g = nflgame.games(yr, week=w)
        ps = nflgame.combine_game_stats(g)
        
        ev_wk_score = []
        ix = 0
        ixp = 0
        found = False
        for p in ps:
            # print("name ",p.name)
            if p.player is None:
                continue         
            else:
                if p.name == pname:
                    ixp = ix
                    found = True
                    p_all_wk_idx.append(ixp)
                else:
                    ev_wk_score.append(score_player(p))
                    ix = ix + 1
        if found == False:
            # handle players that aren't found and assign them worst rank for the week?
            p_all_wk_idx.append(-1)
        everyone_score.append(ev_wk_score)

    print("es len ", len(everyone_score))
    print("es dim "," ".join(map(str,[ len(l) for l in everyone_score]))) # check dimensions
    
    # get rankings
    all_rankings = [np.array(x).argsort() for x in everyone_score]
    print("p_all_wk_idx len", len(p_all_wk_idx))
    # take the ith week of all rankings, and within that list, using the index of the player we found above, find their rank within that week!
    # take the median of these rankings...?
    rnks = []

    # @note to fix!!
    # bad fix for p_all_wk_idx > 7, probably due to duplicate / name matching
    if len(p_all_wk_idx) <= 7:
        lm_rng = len(p_all_wk_idx)
    else:
        lm_rng = 7
    for wi in range(0, lm_rng):

        print("all rankings len", len(all_rankings[wi]), "p_idx_rnk ", p_all_wk_idx[wi])
        plr_idx = p_all_wk_idx[wi]
        if plr_idx < 0:
            # couldn't find player for that week -> assign the worst rank
            rnks.append(max(all_rankings[wi]))
            continue
        rnks.append(all_rankings[wi][plr_idx])
    
    p_rank_wk_med = np.median(rnks)
    return(p_rank_wk_med)              
    

yrs = [2017,2018,2019]

Path("datasets").mkdir(exist_ok=True)
# still need:
# the player’s fantasy score average based on the previous 7 games, 
# the player’s rank based on the previous 7 games

all_dat = []
fn_all_path = "datasets/nflgame_seasons_{}.csv".format("-".join(map(str,yrs)))
for yr in yrs:
    appended_wks = []
    fn_path = "datasets/nflgame_season{}.csv".format(yr)
    # tmp_path = "datasets/temp.csv"
    # print("yr ",yr)
    for wk in range(1,18):
        print("yr ", yr, ", wk ",wk)
        games = nflgame.games(yr, week=wk) # games in x week of y season
        players = nflgame.combine_game_stats(games) # player stats in these games
        # p.player.full_name, 
        temp_dat = []
        if players is None:
            continue
        else:
            for p in players:
                print("name ",p.name)
                if p.player is None:
                    continue            
                else:
                    sga = get_past_player_fantasy_avg(pname = p.name, wk=wk, yr=yr, k=7)
                    rnk = get_past_player_fantasy_rank(pname = p.name, wk=wk, yr=yr, k=7)
                    temp_dat.append([p.name, p.playerid, p.player.full_name, wk, yr, int(p.home), 
                                    p.team, p.player.position, score_player(p), sga,rnk])
        wks_dat = pd.DataFrame(temp_dat,columns = ["p.name", "playerid", "name","wk","yr", "is_home","opp_team","position","score", "7_game_avg", "rank"])
        appended_wks.append(wks_dat)
    print("Writing season to {}".format(fn_path))
    yr_dat = pd.concat(appended_wks)
    yr_dat.to_csv(fn_path)
    all_dat.append(yr_dat)

print("Writing season to {}".format(fn_all_path))
all_dat_cat = pd.concat(all_dat)
all_dat_cat.to_csv(fn_all_path)



# import nflgame
# for week in range(7, 10):
#     games = nflgame.games(2014, week, home='BAL', away='BAL')
#     players = nflgame.combine_game_stats(games)
#     for p in players:
#         if 'S.Smith' in p.name:
#             print(p.player.full_name)
# # get injuries
# response = get("https://www.pro-football-reference.com/players/injuries.htm")
# soup = BeautifulSoup(response.content, 'html.parser')
# table = soup.find('table', {'id': 'results'}) # find the table element in the html, luckily has unique id and res for each row in table 
# df = pd.read_html(str(table))[0]

# df.columns = df.columns.droplevel(level = 0)
# df = df[df['Pos'] != 'Pos']
# fn_path2 = "datasets/injuries_{}.csv".format(datetime.today().strftime('%Y-%m-%d'))
# df.to_csv(fn_path2)
