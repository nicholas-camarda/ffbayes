#!python
# Took this from the internet - this is not mine!! Made some small changes just for me.
# ctrl + alt + R to run this -- alternatively, use Run Python File (drop down from the playbutton on the right in VSCode)

import pandas as pd; pd.set_option('display.max_rows', None)
from bs4 import BeautifulSoup as BS
import requests
from datetime import date
pd.options.mode.chained_assignment = None

# examples and learning
index = ['QB', 'RB', 'WR']

df = pd.DataFrame({
    'A': [20, 19, 15],
    'B': [25, 11, 14]
}, index=index)
df['A - B'] = df['A'] - df['B']

# notice QB scored the most, but didn't have the highest differential, and therefore wasn't the most important player
print(df.head())

# You see, you have a limited amount of spots on your starting to play your players. 
# And your opponent also has a limited number of starting roster spots to play their players. 
# And so the goal of fantasy football is not to maximize how many points you'll score, 
# but maximize your scoring differential at each position in relation to your opponent.

# You can think of these scoring differentials as our value over replacement numbers. 

df['VOR'] = abs(df['A - B'])
df['VOR'].sort_values(ascending=False)

print(df.head())


# this is harder to do with a larger team pool

# What we have to do is find a "replacement player" for each position in the draft pool - 
# a player who's projected points represents the average postional value at each position. 
# Then, with respect to each player's position (this is important. You want to compare each 
# player's projected points to their position's replacement value), substract out the replacement 
# value you calculated from your replacement player. The value you're left with is each player's 
# value over the typical replacement player, or for short, their value over replacement.

### VARIABLES ###
ppr = 0.5
TOP_RNK = 120 # TOP_RNK = 100
### VARIABLES ###

# Algorithm
# 1 - Look at ADP for the upcoming draft year and look at pick #100.
# 2 - Starting from pick 100, go backwards and look for the last WR, RB, QB, and TE picked thus far. 
# These players are your replacement players.

# Our first step is to find our replacement players. 
# We'll find this using ADP data provided by FantasyPros. We have to scrape this data. 
BASE_URL = "https://www.fantasypros.com/nfl/adp/ppr-overall.php"

def make_adp_df():
    res = requests.get(BASE_URL)
    if res.ok:
        soup = BS(res.content, 'html.parser')
        table = soup.find('table', {'id': 'data'}) # found this by looking at the table using "inspect" on the webpage
        df = pd.read_html(str(table))[0]
        print('Output after reading the html:\n\n', df.head(), '\n') # so you can see the output at this point
        df = df[['Player Team (Bye)', 'POS', 'AVG']]
        print('Output after filtering:\n\n', df.head(), '\n')
        df['PLAYER'] = df['Player Team (Bye)'].apply(lambda x: ' '.join(x.split()[:-2])) # removing the team and position
        df['POS'] = df['POS'].apply(lambda x: x[:2]) # removing the position rank
        
        df = df[['PLAYER', 'POS', 'AVG']].sort_values(by='AVG')
        
        print('Final output: \n\n', df.head())
        
        return df
        
    else:
        print('oops, something didn\'t work right', res.status_code)    
adp_df = make_adp_df()
print("ADP df: ", len(adp_df))

# create our dictionary for replacement players, by position
replacement_players = {
    'RB': '',
    'WR': '',
    'TE': '',
    'QB': ''
}

# basically, the last player in this dictionary was the last player chosen for that position out of the 100 players searched
print("\nGenerating replacements...")
for _, row in adp_df[:TOP_RNK].iterrows():
    position = row['POS']
    player = row['PLAYER']
    replacement_players[position] = player

# print(replacement_players.items())
# now that we have replacement players, we need projection data from fantasypros.com

# each position has a different associated URL. We'll create a string format here and loop through the possible positions
BASE_URL = 'https://www.fantasypros.com/nfl/projections/{position}.php?week=draft'
def make_projection_df(ppr):
    
    # we are going to concatenate our individual position dfs into this larger final_df
    final_df = pd.DataFrame()
    
    #url has positions in lower case
    for position in ['rb', 'qb', 'te', 'wr']:
        
        res = requests.get(BASE_URL.format(position=position)) # format our url with the position
        if res.ok:
            soup = BS(res.content, 'html.parser')
            table = soup.find('table', {'id': 'data'})
            df = pd.read_html(str(table))[0]
            
            df.columns = df.columns.droplevel(level=0) # our data has a multi-level column index. The first column level is useless so let's drop it.
            df['PLAYER'] = df['Player'].apply(lambda x: ' '.join(x.split()[:-1])) # fixing player name to not include team
            
            # if you're not doing PPR, don't include this. If you're doing Half PPR,
            # multiply receptions * 1/2
            if 'REC' in df.columns:
                df['FPTS'] = df['FPTS'] + ppr*df['REC'] # add receptions if they're in there. 
            
            df['POS'] = position.upper() # add a position column
            
            df = df[['PLAYER', 'POS', 'FPTS']]
            final_df = pd.concat([final_df, df]) # iteratively add to our final_df
        else:
            print('oops something didn\'t work right', res.status_code)
            return
    
    final_df = final_df.sort_values(by='FPTS', ascending=False) # sort df in descending order on FPTS column
    
    return final_df

print("\nScraping projections dataset...")
df = make_projection_df(ppr = ppr)
print("Saving...")
df.to_csv("/Users/ncamarda/Desktop/coding/ffbayes/datasets/current_projections.csv")
print(len(df), df.head())


replacement_values = {
    'RB': 0,
    'WR': 0,
    'QB': 0,
    'TE': 0
}

# get the points for each replacement player from our projected points dataset
for position, player in replacement_players.items():
    if position in replacement_values.keys():
        replacement_values[position] = df.loc[df['PLAYER'] == player].values[0, -1]
    
print("\nReplacement values: ", replacement_values)

df['VOR'] = df.apply(
    lambda row: row['FPTS'] - replacement_values.get(row['POS']), axis=1
)

print("VOR df: ", df.head())

df = df.sort_values(by='VOR', ascending=False)
df['VALUERANK'] = df['VOR'].rank(ascending=False)
print("\nFinal df: \n", df.head(TOP_RNK))

today = date.today()
this_year = today.strftime("%Y")

fn_to_save = "/Users/ncamarda/Desktop/coding/ffbayes/datasets/snake-draft_ppr-{ppr}_vor_top-{TOP_RNK}_{YEAR}.csv".format(ppr=ppr, TOP_RNK=TOP_RNK, YEAR=this_year)
print(fn_to_save)
df.to_csv(fn_to_save)

print("Done!")