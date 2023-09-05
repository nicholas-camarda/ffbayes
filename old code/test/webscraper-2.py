from requests import get #  HTTP requests straight from Python.
import pandas as pd 
from bs4 import BeautifulSoup as BS
import requests
# If you inspect element on the page, you'll find the data we need is hidden in a table tag with an id of 'data'. 
# If you want to use a different format, go to that URL and toggle the drop down list to your league format. 
# The URL will change, and that will be the URL you will use in your function.
BASE_URL = "https://www.fantasypros.com/nfl/adp/ppr-overall.php"

def make_adp_df():
    res = requests.get(BASE_URL)
    if res.ok:
        soup = BS(res.content, 'html.parser')
        table = soup.find('table', {'id': 'data'})
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
        
df = make_adp_df()


# 	position	week	year	score	is_home	playerid	name	opp_team	team	7_game_avg	rank
# 0	RB	8	2012	21.7	1	00-0025394	Adrian Peterson	TB	MIN	17.81428571	1
# 1	RB	8	2012	4.6	1	00-0027888	Toby Gerhart	TB	MIN	3.214285714	2
