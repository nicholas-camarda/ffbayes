#!python
# copied from https://stmorse.github.io/journal/pfr-scrape-python.html, added adjustments to get this to run + progress bar
import os

# from pro_football_reference_web_scraper import player_game_log as p
# from pro_football_reference_web_scraper import team_game_log as t
import nfl_data_py as nfl
import pandas as pd
from alive_progress import alive_bar

years_to_process = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def create_dataset(year):
    players = nfl.import_weekly_data([year])
    # full_players = players
    # print(full_players.columns)
    # player_id player_display_name position recent_team  season  week season_type  fantasy_points  fantasy_points_ppr
    players_df = players[
        [
            'player_id',
            'player_display_name',
            'position',
            'recent_team',
            'season',
            'week',
            'season_type',
            'fantasy_points',
            'fantasy_points_ppr',
        ]
    ]
    players = players.rename(
        columns={'recent_team': 'player_team', 'season_type': 'game_type'},
    )

    schedules = nfl.import_schedules([year])
    # full_schedules = schedules
    #  game_id  week game_type home_team away_team  away_score  home_score
    schedules_df = schedules[
        [
            'game_id',
            'week',
            'season',
            'gameday',
            'game_type',
            'home_team',
            'away_team',
            'away_score',
            'home_score',
        ]
    ]
    # schedules = schedules.rename(columns={"gsis":"gsis_id"})
    # full_players.to_csv("~/Downloads/players.csv")
    # full_schedules.to_csv("~/Downloads/schedules.csv")

    # Step 2: Merge the dataframes

    # Merging considering players as part of the home team
    home_merge = players_df.merge(
        schedules_df,
        left_on=['season', 'week', 'recent_team'],
        right_on=['season', 'week', 'home_team'],
        how='left',
    )

    # Merging considering players as part of the away team
    away_merge = players_df.merge(
        schedules_df,
        left_on=['season', 'week', 'recent_team'],
        right_on=['season', 'week', 'away_team'],
        how='left',
    )

    # Step 3: Create columns to identify home and away players
    home_merge['is_home_team'] = home_merge['home_team'].notna()
    away_merge['is_away_team'] = away_merge['away_team'].notna()

    # Step 4: Concatenate the dataframes
    merged_df = pd.concat([home_merge, away_merge])

    # Step 5: Retain specific columns
    final_columns1 = [
        'player_id',
        'player_display_name',
        'position',
        'recent_team',
        'home_team',
        'season',
        'week',
        'game_type',
        'fantasy_points',
        'fantasy_points_ppr',
        'game_id',
        'gameday',
        'away_team',
        'away_score',
        'home_score',
    ]
    final_df1 = merged_df[final_columns1]

    # Step 6: Remove rows with NaN game_id and filter rows
    final_df1 = final_df1[final_df1['game_id'].notna()]
    final_df1['player_team'] = final_df1['recent_team']
    final_df1 = final_df1[
        (
            (final_df1['player_team'] == final_df1['home_team'])
            | (final_df1['player_team'] == final_df1['away_team'])
        )
    ]

    # Step 7: Deduplication
    final_df1 = final_df1.drop_duplicates()
    final_df1

    # Step 8: Get injuries and merge
    injuries = nfl.import_injuries([year])
    # full_injuries = injuries
    # full_injuries.to_csv("~/Downloads/injuries.csv")
    injuries = injuries[
        [
            'full_name',
            'position',
            'week',
            'season',
            'team',
            'game_type',
            'report_status',
            'practice_status',
        ]
    ]
    injuries = injuries.rename(
        columns={'full_name': 'player_display_name', 'team': 'home_team'},
    )

    # Merge the merged data with the injuries data
    final_df = pd.merge(
        final_df1,
        injuries,
        on=[
            'player_display_name',
            'position',
            'season',
            'week',
            'game_type',
            'home_team',
        ],
        how='left',
    )
    final_df = final_df.rename(
        columns={
            'report_status': 'game_injury_report_status',
            'practice_status': 'practice_injury_report_status',
        },
    )

    return final_df


def scrape_ff_data_by_year(year=[2018, 2019]):
    print('Processing year...')

    final_df = create_dataset(year)
    maxp = len(final_df)
    data_list = []

    with alive_bar(maxp) as bar:
        for i, row in final_df.iterrows():
            # Get the player details
            player_id = row['player_id']
            player_name = row['player_display_name']
            season = row['season']
            position = row['position']
            week = row['week']
            fantasy_points_ppr = row['fantasy_points_ppr']
            fantasy_points = row['fantasy_points']

            game_date = row['gameday']
            team = row[
                'recent_team'
            ]  # Assuming 'recent_team' contains the team information
            home = row['home_team']
            away = row[
                'away_team'
            ]  # Assuming 'away_team' contains the away team information
            opponent = away if team == away else home  # Get the opponent team
            game_injury_report_status = row['game_injury_report_status']
            practice_injury_report_status = row['practice_injury_report_status']
            # Append the data to the data_list
            data_list.append(
                [
                    week,
                    game_date,
                    team,
                    away,
                    opponent,
                    fantasy_points,
                    fantasy_points_ppr,
                    player_name,
                    player_id,
                    position,
                    season,
                    game_injury_report_status,
                    practice_injury_report_status,
                ],
            )
            bar()

    df = pd.DataFrame(
        data_list,
        columns=[
            'G#',
            'Date',
            'Tm',
            'Away',
            'Opp',
            'FantPt',
            'FantPtPPR',
            'Name',
            'PlayerID',
            'Position',
            'Season',
            'GameInjuryStatus',
            'PracticeInjuryStatus',
        ],
    )
    # print("Final dataframe has dimensions: {shape}".format(shape=df.shape))
    print('Saving df...')
    df.to_csv(f'datasets/{year}season.csv')
    print('Done.')

    return df


def combine_datasets(directory_path, output_directory_path, years_to_process):
    # Step 1: Specify the directory containing the CSV files

    # Step 2: Get a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Step 3: Initialize an empty list to hold dataframes
    dataframes = []

    # Step 4: Loop through each CSV file and append its data to the dataframes list
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        dataframes.append(pd.read_csv(file_path))

    # Step 5: Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True).sort_values(by = 'Season')

    # Step 6: Save the combined dataframe to a new CSV file
    combined_df.to_csv(
        f'{output_directory_path}/{years_to_process[0]}-{years_to_process[-1]}combined_data.csv',
        index=False,
    )
    print(f'Check {output_directory_path} for combined dataset')


def main(process_years = years_to_process):
    # make the directory structure
    os.makedirs('datasets', exist_ok=True)
    existing_files = os.listdir('datasets')

    # RUN
    for yr in process_years:
        if any(file.startswith(str(yr)) for file in existing_files):
            print(f'Skipping year {yr} as file already exists')
            continue
        print(f'Processing year {yr}')
        scrape_ff_data_by_year(year=yr)

    combine_datasets('datasets', 'combined_datasets', process_years)
    print('Done with everything!')


if __name__ == '__main__':
    main(years_to_process)
