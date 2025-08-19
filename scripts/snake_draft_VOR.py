#!python
# Took this from the internet - this is not mine!! Made some small changes just for me.
# ctrl + alt + R to run this -- alternatively, use Run Python File (drop down from the playbutton on the right in VSCode)

from io import StringIO

import pandas as pd

pd.set_option('display.max_rows', None)
from datetime import date

import requests
from bs4 import BeautifulSoup as BS

pd.options.mode.chained_assignment = None

# examples and learning
index = ['QB', 'RB', 'WR']

df = pd.DataFrame({'A': [20, 19, 15], 'B': [25, 11, 14]}, index=index)
df['A - B'] = df['A'] - df['B']

print(
    '''
    Teams on column, e.g. Team A and Team B. Positions on rows. 
    Notice QB scored the most, but didn't have the highest differential, and therefore wasn't the most important player
    ''',
)

print(
    '''
    You see, you have a limited amount of spots on your starting to play your players. 
    And your opponent also has a limited number of starting roster spots to play their players. 
    And so the goal of fantasy football is not to maximize how many points you'll score, 
    but maximize your scoring differential at each position in relation to your opponent.
    ''',
)

print(
    'You can think of these scoring differentials as our value over replacement numbers.',
)

df['VOR'] = abs(df['A - B'])
df['VOR'].sort_values(ascending=False)

print(df.head())


# this is harder to do with a larger team pool

print(
    '''
    What we have to do is find a "replacement player" for each position in the draft pool -
    a player who's projected points represents the average postional value at each position.
    Then, with respect to each player's position (this is important. You want to compare each
    player's projected points to their position's replacement value), substract out the replacement
    value you calculated from your replacement player. The value you're left with is each player's
    value over the typical replacement player, or for short, their value over replacement.
''',
)

### VARIABLES ###
ppr = 0.5
TOP_RNK = 120  # TOP_RNK = 100
### VARIABLES ###

# print(
#     """
#     The Algorithm works like this:
#     #1 - Look at ADP (average draft position) for the upcoming draft year and look at pick #100 (for example, but this is a modifiable value).
#     #2 - Starting from pick 100, go backwards and look for the last WR, RB, QB, and TE picked thus far.
#     These players are your replacement players.
#     """
# )

# Our first step is to find our replacement players.
# We'll find this using ADP data provided by FantasyPros. We have to scrape this data.
BASE_URL = 'https://www.fantasypros.com/nfl/adp/ppr-overall.php'


def make_adp_df():
    res = requests.get(BASE_URL)
    if res.ok:
        soup = BS(res.content, 'html.parser')
        table = soup.find(
            'table', {'id': 'data'},
        )  # found this by looking at the table using "inspect" on the webpage
        df = pd.read_html(StringIO(str(table)))[0]  # pd.read_html(str(table))[0]
        print(
            'Output after reading the html:\n\n', df.head(), '\n',
        )  # so you can see the output at this point
        df = df[['Player Team (Bye)', 'POS', 'AVG']]
        print('Output after filtering:\n\n', df.head(), '\n')
        df['PLAYER'] = df['Player Team (Bye)'].apply(
            lambda x: ' '.join(x.split()[:-2]),
        )  # removing the team and position
        df['POS'] = df['POS'].apply(lambda x: x[:2])  # removing the position rank

        df = df[['PLAYER', 'POS', 'AVG']].sort_values(by='AVG')

        print('Final output: \n\n', df.head())

        return df

    print("oops, something didn't work right", res.status_code)


adp_df = make_adp_df()
print('ADP df: ', len(adp_df))

# create our dictionary for replacement players, by position
replacement_players = {'RB': '', 'WR': '', 'TE': '', 'QB': ''}

# basically, the last player in this dictionary was the last player chosen for that position out of the 100 players searched
print('\nGenerating replacements...')
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

    # url has positions in lower case
    for position in ['rb', 'qb', 'te', 'wr']:
        res = requests.get(
            BASE_URL.format(position=position),
        )  # format our url with the position
        if res.ok:
            soup = BS(res.content, 'html.parser')
            table = soup.find('table', {'id': 'data'})
            df = pd.read_html(StringIO(str(table)))[0]  # pd.read_html(str(table))[0]

            df.columns = df.columns.droplevel(
                level=0,
            )  # our data has a multi-level column index. The first column level is useless so let's drop it.
            df['PLAYER'] = df['Player'].apply(
                lambda x: ' '.join(x.split()[:-1]),
            )  # fixing player name to not include team

            # if you're not doing PPR, don't include this. If you're doing Half PPR,
            # multiply receptions * 1/2
            if 'REC' in df.columns:
                df['FPTS'] = (
                    df['FPTS'] + ppr * df['REC']
                )  # add receptions if they're in there.

            df['POS'] = position.upper()  # add a position column

            df = df[['PLAYER', 'POS', 'FPTS']]
            final_df = pd.concat([final_df, df])  # iteratively add to our final_df
        else:
            print("oops something didn't work right", res.status_code)
            return None

    final_df = final_df.sort_values(
        by='FPTS', ascending=False,
    )  # sort df in descending order on FPTS column

    return final_df


def run_pipeline(ppr_value = ppr, top_rank = TOP_RNK):
    print('\nScraping projections dataset...')
    df = make_projection_df(ppr=ppr_value)
    print('Saving...')
    df.to_csv('misc-datasets/current_projections.csv')
    print(len(df), df.head())
    replacement_values = {'RB': 0, 'WR': 0, 'QB': 0, 'TE': 0}

    # get the points for each replacement player from our projected points dataset
    for position, player in replacement_players.items():
        if position in replacement_values:
            replacement_values[position] = df.loc[df['PLAYER'] == player].values[0, -1]

    print('\nReplacement values: ', replacement_values)

    df['VOR'] = df.apply(
        lambda row: row['FPTS'] - replacement_values.get(row['POS']), axis=1,
    )

    print('VOR df: ', df.head())

    df = df.sort_values(by='VOR', ascending=False)
    df['VALUERANK'] = df['VOR'].rank(ascending=False)
    print('\nFinal df: \n', df.head(top_rank))

    today = date.today()
    this_year = today.strftime('%Y')

    fn_to_save = (
        f'snake_draft_datasets/snake-draft_ppr-{ppr_value}_vor_top-{top_rank}_{this_year}.csv'
    )
    print(fn_to_save)
    df.to_csv(fn_to_save)

    print('Done!')
    return df, fn_to_save


def generate_draft_strategy(dataframe, output_file_path):
    # Function to categorize players into early, mid, and late draft phases based on their VALUERANK
    def categorize_draft_phase(row):
        if row['VALUERANK'] <= 70:
            return 'Early Draft'
        if 70 < row['VALUERANK'] <= 140:
            return 'Mid Draft'
        return 'Late Draft'

    # Applying the function to create a new column 'Draft Phase'
    dataframe['Draft_Phase'] = dataframe.apply(categorize_draft_phase, axis=1)

    # Creating lists of players to target in each draft phase
    early_draft_targets = dataframe[
        dataframe['Draft_Phase'] == 'Early Draft'
    ].sort_values(by='VOR', ascending=False)
    mid_draft_targets = dataframe[dataframe['Draft_Phase'] == 'Mid Draft'].sort_values(
        by='VOR', ascending=False,
    )
    late_draft_targets = dataframe[
        dataframe['Draft_Phase'] == 'Late Draft'
    ].sort_values(by='VOR', ascending=False)

    # Draft strategy for each round
    draft_strategy = {
        'Round': list(range(1, 17)),
        'Primary Target': [
            'Top-tier RB or WR (whichever has the highest VOR available)',
            'Another top-tier RB or WR (focus on filling the position not filled in Round 1)',
            'RB or WR (focus on acquiring high VOR players)',
            'RB or WR (aim to have at least two strong players in both positions by the end of this round)',
            'RB or WR (continue to build depth at these positions)',
            'Mid-tier QBs and TEs if not selected in earlier rounds',
            'Mid-tier QBs and TEs if not selected in earlier rounds',
            'Focus on building depth at RB and WR positions',
            'Focus on building depth at RB and WR positions',
            'Top Defense/Special Teams unit',
            'Draft high-upside players (potential breakout candidates) at RB and WR positions',
            'Draft high-upside players (potential breakout candidates) at RB and WR positions',
            'Draft high-upside players (potential breakout candidates) at RB and WR positions',
            'Defense/Special Teams (if not selected in middle rounds)',
            'Kicker',
            'Kicker (if not selected in Round 15)',
        ],
        'Backup Plan': [
            'If top-tier RBs and WRs are taken, consider a top-tier TE',
            'If a top-tier TE is still available, this might be a good time to secure that position',
            'Consider drafting a top-tier QB if available',
            'QB (if not already selected in previous rounds)',
            'TE (if not already selected in previous rounds)',
            'Continue to build depth at RB and WR positions',
            'Continue to build depth at RB and WR positions',
            'Consider drafting a top Defense/Special Teams unit',
            'Consider drafting a top Defense/Special Teams unit',
            'Focus on building depth at RB and WR positions',
            'Consider drafting a backup QB or TE for bench depth',
            'Consider drafting a backup QB or TE for bench depth',
            'Consider drafting a backup QB or TE for bench depth',
            'High-upside players to fill out bench depth',
            'High-upside players at RB and WR positions',
            'High-upside players at RB and WR positions',
        ],
    }

    draft_strategy_df = pd.DataFrame(draft_strategy)

    # Creating an Excel writer object
    with pd.ExcelWriter(output_file_path) as writer:
        # Writing data to different sheets based on the draft phase
        early_draft_targets.to_excel(
            writer, sheet_name='Early Draft (Rounds 1-5)', index=False,
        )
        mid_draft_targets.to_excel(
            writer, sheet_name='Mid Draft (Rounds 6-10)', index=False,
        )
        late_draft_targets.to_excel(
            writer, sheet_name='Late Draft (Rounds 11-16)', index=False,
        )

        # Writing the draft strategy for each round
        draft_strategy_df.to_excel(
            writer, sheet_name='Round-by-Round Draft Strategy', index=False,
        )

        # Writing a summary sheet with the strategy
        summary_data = {
            'Draft Phase': [
                'Early Draft (Rounds 1-5)',
                'Mid Draft (Rounds 6-10)',
                'Late Draft (Rounds 11-16)',
            ],
            'Primary Focus': [
                'Top WRs and RBs with the highest VOR values.',
                'Mid-tier WRs and RBs who still offer good value.',
                'High-upside players who could potentially break out.',
            ],
            'Secondary Focus': [
                'A top QB or TE if available.',
                'Start filling out other positions like QB and TE if not already picked in early rounds.',
                'Backup players and streaming options for positions like defense and kicker.',
            ],
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Draft Strategy Summary', index=False)


def main():
    df, fn_to_save = run_pipeline()
    today = date.today()
    this_year = today.strftime('%Y')
    dataframe = pd.read_csv(fn_to_save)
    generate_draft_strategy(
        dataframe,
        f'snake_draft_datasets/DRAFTING STRATEGY -- snake-draft_ppr-{ppr}_vor_top-{TOP_RNK}_{this_year}.xlsx',
    )


if __name__ == '__main__':
    main()
