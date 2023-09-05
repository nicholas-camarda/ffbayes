#!python
# copied from https://stmorse.github.io/journal/pfr-scrape-python.html, added adjustments to get this to run + progress bar
import os
import time
from datetime import datetime
from pathlib import Path
from sys import argv

import numpy as np
import pandas as pd
import requests  # HTTP requests straight from Python.
from alive_progress import alive_bar
from bs4 import BeautifulSoup  # parse the raw html
from pro_football_reference_web_scraper import player_game_log as p
from requests.exceptions import HTTPError

years_to_process = [2017, 2018, 2019, 2020, 2021, 2022, 2023]


def scrape_ff_data_by_year(year=2018):
    url = "https://www.pro-football-reference.com"
    full_url = url + "/years/" + str(year) + "/fantasy.htm"
    print(full_url)
    # grab fantasy players
    try:
        r = requests.get(full_url)
        r.raise_for_status()
    except HTTPError as e:
        if e.response.status_code == 429:
            print("HTTP 429 error received. Retrying in 10 seconds...")
            time.sleep(10)
            r = requests.get(full_url)
        else:
            raise

    soup = BeautifulSoup(r.content, "html.parser")
    tables = soup.find_all("table")
    if len(tables) == 0:
        print(f"No tables found for year {year}. Skipping...")
        return
    parsed_table = tables[0]  # .find_all("table")[0]
    print(parsed_table)
    lst_to_iterate = parsed_table.find_all("tr")[2:]

    maxp = len(lst_to_iterate)

    # first 2 rows are col headers
    print("Processing scraped table...")
    df = []
    with alive_bar(len(lst_to_iterate)) as bar:
        for i, row in enumerate(lst_to_iterate):
            # if i % 25 == 0: print(i)
            if i >= maxp:
                print("\nComplete.")
                break

            try:
                dat = row.find("td", attrs={"data-stat": "player"})

                name = dat.a.get_text()  # hyperlink
                stub = dat.a.get("href")
                # hyperlink's destination for the player, which contains all their stats
                # - these both may fail, which is why the catch is in place
                stub = stub[:-4] + "/fantasy/" + str(year)
                pos = row.find("td", attrs={"data-stat": "fantasy_pos"}).get_text()

            except AttributeError as ae:
                print("Link anchor does not exist.")
                pass

            # grab this players stats
            try:
                tdf = pd.read_html(url + stub)[0]
            except HTTPError as e:
                if e.response.status_code == 429:
                    print("HTTP 429 error received. Retrying in 10 seconds...")
                    time.sleep(10)
                    tdf = pd.read_html(url + stub)[0]
                else:
                    raise
            copy_tdf = tdf.copy(
                deep=True
            )  # must deep copy before replacing these values, so as to not chain link

            # there are some players that got entries but never played, so filter out all player data tables with 1 or less rows
            if not copy_tdf.empty and len(copy_tdf) > 1:
                # get rid of MultiIndex, just keep last row
                copy_tdf.columns = copy_tdf.columns.get_level_values(-1)

                # adjust the away/home column
                copy_tdf_renamed = copy_tdf.rename(
                    columns={"Unnamed: 4_level_2": "Away"}
                )
                # print(tdf.columns)

                away_fix = []
                for r in copy_tdf_renamed.loc[:, "Away"]:
                    if r == "@":
                        # away
                        away_fix.append(1)
                    else:
                        # home
                        away_fix.append(0)

                copy_tdf_renamed.loc[:, "Away"] = away_fix

                # drop all intermediate stats
                copy_tdf_renamed_dropped_1 = copy_tdf_renamed.iloc[
                    :, [1, 2, 3, 4, 5, -3]
                ]

                # drop "Total" row
                copy_tdf_final = copy_tdf_renamed_dropped_1.query('Date != "Total"')

                copy2_tdf_final = copy_tdf_final.copy(deep=True)
                # add other info
                copy2_tdf_final.loc[:, "Name"] = name
                copy2_tdf_final.loc[:, "Position"] = pos
                copy2_tdf_final.loc[:, "Season"] = year

                df.append(copy2_tdf_final)
            else:
                pass
            bar()

    df = pd.concat(df)
    print("Final dataframe has dimensions: {shape}".format(shape=df.shape))
    print("Saving df...")
    df.to_csv("datasets/{yr}season.csv".format(yr=year))
    print("Done.")


existing_files = os.listdir(
    "datasets"
)  # Replace with the actual path to your directory

for yr in years_to_process:
    if any(file.startswith(str(yr)) for file in existing_files):
        print(f"Skipping year {yr} as file already exists")
        continue
    print(f"Processing year {yr}")
    scrape_ff_data_by_year(year=yr)

print("Done with everything!")
