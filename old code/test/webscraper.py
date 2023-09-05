# https://www.fantasyfootballdatapros.com/blog/intermediate/4

from sys import argv
from requests import get #  HTTP requests straight from Python.
import pandas as pd 
from bs4 import BeautifulSoup # parse the raw html


year = input('What season? Note: Input a season between 1999 and 2019. If not I cannot help you :) ')
week = input('What week of the {0} season? '.format(year))

year, week = int(year), int(week)

# pro-football-ref has a difference webpage for each stat.. need to merge later

passingURL = """
https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game&year_min={year}&year_max={year}&season_start=1&season_end=-1&age_min=0&age_max=99&game_type=A&league_id=&team_id=&opp_id=&game_num_min=0&game_num_max=99&week_num_min={week}&week_num_max={week}&game_day_of_week=&game_location=&game_result=&handedness=&is_active=&is_hof=&c1stat=pass_att&c1comp=gt&c1val=1&c2stat=&c2comp=gt&c2val=&c3stat=&c3comp=gt&c3val=&c4stat=&c4comp=gt&c4val=&order_by=pass_rating&from_link=1
""".format(year=year, week=week)

receivingURL = """
https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game&year_min={year}&year_max={year}&season_start=1&season_end=-1&age_min=0&age_max=99&game_type=A&league_id=&team_id=&opp_id=&game_num_min=0&game_num_max=99&week_num_min={week}&week_num_max={week}&game_day_of_week=&game_location=&game_result=&handedness=&is_active=&is_hof=&c1stat=rec&c1comp=gt&c1val=1&c2stat=&c2comp=gt&c2val=&c3stat=&c3comp=gt&c3val=&c4stat=&c4comp=gt&c4val=&order_by=rec_yds&from_link=1
""".format(year=year, week=week)

rushingURL = """
https://www.pro-football-reference.com/play-index/pgl_finder.cgi?request=1&match=game&year_min={year}&year_max={year}&season_start=1&season_end=-1&age_min=0&age_max=99&game_type=A&league_id=&team_id=&opp_id=&game_num_min=0&game_num_max=99&week_num_min={week}&week_num_max={week}&game_day_of_week=&game_location=&game_result=&handedness=&is_active=&is_hof=&c1stat=rush_att&c1comp=gt&c1val=1&c2stat=&c2comp=gt&c2val=&c3stat=&c3comp=gt&c3val=&c4stat=&c4comp=gt&c4val=&order_by=rush_yds&from_link=1
""".format(year=year, week=week)


urls = {
    'Passing': passingURL,
    'Receiving': receivingURL,
    'Rushing': rushingURL
}

dfs = [] # to merge the 3 df's 

defColumnSettings = {
    'axis': 1,
    'inplace': True
}

# simply a dictionary of keyword arguments we’ll be passing to pandas multiple times. 
# defColumnSettings is shorthand for default column settings. 
# If you’re not super familiar with Python, you can pass in keyword arguments to 
# a function with a dictionary using a special notation (double asterisks). 
# In short, these two lines of code below are exactly equivalent.

# Since we’ll be changing a lot of columns, we’ll be setting axis = 1 
# and inplace = True a lot, so I simply saved the keyword arguments to a dictionary,

dfs.drop('ColumnName', axis=1, inplace=True)
# df.drop('ColumnName', **defColumnSettings)


for key, url in urls.items():

    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'results'}) # find the table element in the html, luckily has unique id and res for each row in table 

    df = pd.read_html(str(table))[0]

    df.columns = df.columns.droplevel(level = 0)

    df.drop(['Result', 'Week', 'G#', 'Opp', 'Unnamed: 7_level_1', 'Age', 'Rk', 'Lg', 'Date', 'Day'], **defColumnSettings)

    df = df[df['Pos'] != 'Pos']

    df.set_index(['Player', 'Pos', 'Tm'], inplace=True)

    if key == 'Passing':
        df = df[['Yds', 'TD', 'Int', 'Att', 'Cmp']]
        df.rename({'Yds': 'PassingYds', 'Att': 'PassingAtt', 'Y/A': 'Y/PassingAtt', 'TD': 'PassingTD'}, **defColumnSettings)
    elif key =='Receiving':
        df = df[['Rec', 'Tgt', 'Yds', 'TD']]
        df.rename({'Yds': 'ReceivingYds', 'TD': 'ReceivingTD'}, **defColumnSettings)
    elif key == 'Rushing':
        df.drop('Y/A', **defColumnSettings)
        df.rename({'Att': 'RushingAtt', 'Yds': 'RushingYds', 'TD': 'RushingTD'}, **defColumnSettings)
    dfs.append(df)


df = dfs[0].join(dfs[1:], how='outer')
df.fillna(0, inplace=True)
df = df.astype('int64')

# calculate fantasy points
df['FantasyPoints'] = df['PassingTds']/25 + df['PassingTD']*4 - df['Int']*2 + df['Rec'] + df['ReceivingYds']/10 + df['ReceivingTD']*6 + df['RushingYds']/10 + df['RushingTD']*6
df.reset_index(inplace=True)


try:
    if argv[1] == '--save':
        df.to_csv('datasets/season{}week{}.csv'.format(year, week))
except IndexError:
    pass

# python webscraping.py --save