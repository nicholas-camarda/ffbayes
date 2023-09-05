# https://stmorse.github.io/journal/pfr-scrape-python.html
import pandas as pd
from bs4 import BeautifulSoup
import os
import requests
# import nflgame

url = 'https://www.pro-football-reference.com'
# year = 2018
maxp = 300


# grab fantasy players
r = requests.get(url + '/years/' + str(yr) + '/fantasy.htm')
soup = BeautifulSoup(r.content, 'html.parser')
parsed_table = soup.find_all('table')[0]  

df = []

# first 2 rows are col headers
for i,row in enumerate(parsed_table.find_all('tr')[2:]):
    if i % 10 == 0: print(i, end=' ')
    if i >= maxp: 
        print('\nComplete.')
        break
        
    try:
        dat = row.find('td', attrs={'data-stat': 'player'})
        name = dat.a.get_text()
        stub = dat.a.get('href')
        stub = stub[:-4] + '/fantasy/' + str(yr)
        pos = row.find('td', attrs={'data-stat': 'fantasy_pos'}).get_text()

        # grab this players stats
        tdf = pd.read_html(url + stub)[0]    

        # get rid of MultiIndex, just keep last row
        tdf.columns = tdf.columns.get_level_values(-1)

        # fix the away/home column
        tdf = tdf.rename(columns={'Unnamed: 4_level_2': 'Away'})
        tdf['Away'] = [1 if r=='@' else 0 for r in tdf['Away']]

        # drop all intermediate stats
        tdf = tdf.iloc[:,[1,2,3,4,5,-3]]
        
        # drop "Total" row
        tdf = tdf.query('Date != "Total"')
        
        # add other info
        tdf.loc[:,'Name'] = name
        tdf.loc[:,'Position'] = pos
        tdf.loc[:,'Season'] = yr

        df.append(tdf)
    except:
        pass
    
df = pd.concat(df)
df.head()



