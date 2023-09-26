import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from urllib.parse import urljoin
from utils.parse_requests import request_download, request_and_parse
from utils.file_manipulation import save_data

FOOTBALL_DATA_URL = "https://www.football-data.co.uk"
SLEEP_INTERVAL = 2

CONTAINED_DATASET_COUNTRIES_DATA =  [
        {'name': 'argentina', 'code': 'arg'}, {'name': 'austria', 'code': 'aut'}, 
        {'name': 'brazil', 'code': 'bra'}, {'name': 'china', 'code': 'chn'}, 
        {'name': 'denmark', 'code': 'dnk'}, {'name': 'finland', 'code': 'fin'},
        {'name': 'ireland', 'code': 'irl'}, {'name': 'japan', 'code': 'jpn'},
        {'name': 'mexico', 'code': 'mex'}, {'name': 'norway', 'code': 'nor'}, 
        {'name': 'poland', 'code': 'pol'}, {'name': 'romania', 'code': 'rou'},
        {'name': 'russia', 'code': 'rus'}, {'name': 'sweden', 'code': 'swe'},
        {'name': 'switzerland', 'code': 'swz'}, {'name': 'usa', 'code': 'usa'}
    ]
SEPARATED_DATASET_COUNTRIES_DATA = [
         {'name': 'england', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'championship', 'data name': 'Championship'}, 
             {'file_name': 'premier_league', 'data name': 'Premier League'}, 
             {'file_name': 'conference', 'data name': 'Conference'}, 
             {'file_name': 'league_1', 'data name': 'League 1'}, 
             {'file_name': 'league_2', 'data name': 'League 2'}, 
             {'file_name': 'division_1', 'data name': 'Championship'}, 
             {'file_name': 'division_2', 'data name': 'League 1'}, 
             {'file_name': 'division_3', 'data name': 'League 2'}
         ]}, 
         {'name': 'belgium', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'jupiler_league', 'data name': 'Jupiler League'}
         ]}, 
         {'name': 'france', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'le_championnat', 'data name': 'Le Championnat'}, 
             {'file_name': 'division_2', 'data name': 'Division 2'}
         ]}, 
         {'name': 'germany', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'bundesliga_1', 'data name': 'Bundesliga 1'}, 
             {'file_name': 'bundesliga_2', 'data name': 'Bundesliga 2'}
         ]}, 
         {'name': 'greece', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'ethniki_katigoria', 'data name': 'Ethniki Katigoria'}
         ]}, 
         {'name': 'italy', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'serie_b', 'data name': 'Serie B'}, 
             {'file_name': 'serie_a', 'data name': 'Serie A'}
         ]}, 
         {'name': 'netherlands', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'eredivisie', 'data name': 'Eredivisie'}
         ]}, 
         {'name': 'portugal', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'liga_i', 'data name': 'Liga I'}
         ]}, 
         {'name': 'scotland', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'premier_league', 'data name': 'Premier League'}, 
             {'file_name': 'division_1', 'data name': 'Division 1'}, 
             {'file_name': 'division_2', 'data name': 'Division 2'}, 
             {'file_name': 'division_3', 'data name': 'Division 3'}
         ]}, 
         {'name': 'spain', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'la_liga_primera_division', 'data name': 'La Liga Primera Division'}, 
             {'file_name': 'la_liga_segunda_division', 'data name': 'La Liga Segunda Division'}
         ]}, 
         {'name': 'turkey', 'all_seasons': 'separated', 'leagues': [
             {'file_name': 'futbol_ligi_1', 'data name': 'Futbol Ligi 1'}
         ]}
    ]



def scrape_football_data(main_folder_path: str, update_only: bool=False, seasons: list[str]=["2324"], countries=["england"]) -> None:
    """
    Scraps football data from https://www.football-data.co.uk
    
    Params
    ------
    main_folder_path: folder path of where the files will be downloaded to.
    update_only: if True, only the data for the specified `season` is downloaded. some categories, eg usa, aren't grouped into seasons. These categories have one csv file containing all the seasons grouped together. if update_only, these files are downloaded too.
    season: the season to download data for. Only used if update_only is specified.

    """
    soup = request_and_parse(urljoin(FOOTBALL_DATA_URL, "data.php"))
    tags_to_scrap = soup.select("td > a")
    for link in tags_to_scrap:
        result_name = link.get_text()
        # Check if "Results" is not in the result_name
        if "Results" not in result_name:
            continue

        country_name = result_name.lower().split(" ")[0]
        
        if update_only and (country_name not in countries):
            continue
                
        result_link = urljoin(FOOTBALL_DATA_URL, link.get("href"))
        soup = request_and_parse(result_link)
        time.sleep(SLEEP_INTERVAL) # sleep for 2 seconds so we don't send too many requests at a time to host

        results_page_links = soup.select("a")
        for link in results_page_links:
            datalink = link.get("href")

            # Check if the link is not a CSV file
            if ".csv" not in datalink:
                continue

            datalink_split = datalink.lower().split("/")
            yr = datalink_split[1] if len(datalink_split) == 3 else "all"
            filename = link.get_text().lower().replace(" ",\
                        "_") + ".csv" if len(datalink_split) == 3 else datalink_split[-1].replace(" ", "_")
            
#             yr format - 2324 which stands for the football season considered
#           download only data for specified season. files with yr == all are excluded from this condition
            if update_only and (yr not in seasons and yr != "all"):
                continue
 
            # Request and download the CSV data
            res = request_download(urljoin(FOOTBALL_DATA_URL, datalink))

            # Save the data to the specified folder
            save_data(res, filename, os.path.join(main_folder_path, country_name, yr))
            time.sleep(SLEEP_INTERVAL)
            print(f"Downloaded: {country_name} - {yr} - {filename}")


def get_contained_dataset() -> pd.DataFrame:
    """Returns a dataset containing all the data in contained datasets concatenated together"""
    contained_datasets = []
    # Time seems like a good feature but has been removed cause other datasets in separated_dataset do not include it until 1920 season     
    columns = ["Country", "League", "Season", "Date", "Home", "Away", "HG", "AG", "Res", "AvgH", "AvgD", "AvgA"]
    for country_data in CONTAINED_DATASET_COUNTRIES_DATA:
        dataset_path = os.path.join("Datasets", "football_results",  country_data["name"], "all", country_data["code"] + ".csv")     
        dataset = pd.read_csv(dataset_path, usecols=columns, dayfirst=True, parse_dates=["Date"])
        contained_datasets.append(dataset)
    # concatenate and return dataset, sorted by date-time
    return pd.concat(contained_datasets).set_index(["Date"]).sort_index().reset_index()

def get_separated_dataset() -> pd.DataFrame:
    """Returns a dataset containing all the data in separated datasets concatenated together.
       Separated datasets are datasets grouped categorically into seasons.
    """
    columns = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','IWH','IWD','IWA', "HT", "AT", "SBD", "SBA", "SBH", "WHH", "WHD", "WHA"]
    separated_datasets = []
    for country_data in SEPARATED_DATASET_COUNTRIES_DATA:
        start_year = "2000" # results from this year up till the current year is what we'll use for our models
        for i in pd.date_range(start_year, str(datetime.now().year + 1 ), freq='1Y'):
            current_year = i.year
            next_year = current_year + 1
            # seasons are in the form of 2324 where 23 is current year and 24 is next year
            season = str(current_year)[-2:] + str(next_year)[-2:]

            for league in country_data["leagues"]:
                country_name = country_data["name"]
                file_name = league["file_name"] + ".csv"
                dataset_path = os.path.join("Datasets", "football_results",  country_name, season, file_name) 
                try:
                    dataset = pd.read_csv(dataset_path, usecols=lambda x: x in columns)
                    dataset["Date"] = pd.to_datetime(dataset["Date"], dayfirst=True, format="mixed")
                except FileNotFoundError:
                    print("=" * 100)
                    print(f"Could not find {dataset_path}")
                    print("=" * 100)
                    continue
                except UnicodeDecodeError as err:
                    try:
                        dataset = pd.read_csv(dataset_path, usecols=lambda x: x in columns, \
                                              encoding='ANSI', on_bad_lines='skip')
                        dataset["Date"] = pd.to_datetime(dataset["Date"], dayfirst=True, format="mixed")
                    except Exception as err:
                        print(f"{err}: {dataset_path}")
                        print("=**" * 33)
                        continue
                except ValueError as err:
                    print(f"{err}, {dataset_path}")
                dataset.rename(columns={"HomeTeam": "Home", "AwayTeam": "Away", "FTHG": "HG", "FTAG": "AG", "FTR": "Res", "HT": "Home", "AT": "Away"}, inplace=True)

                if ("SBD" in dataset.columns) and ("IWH" in dataset.columns):
                    dataset["AvgH"] = (dataset["IWH"] + dataset["SBH"]) / 2
                    dataset["AvgD"] = (dataset["IWD"] + dataset["SBD"]) / 2
                    dataset["AvgA"] = (dataset["IWA"] + dataset["SBA"]) / 2 
                    dataset.drop(columns=["IWH", "IWD", "IWA", "SBH", "SBD", "SBA"], inplace=True)
                else:
                    if "IWH" in dataset.columns:
                        dataset["AvgH"] = dataset["IWH"]
                        dataset["AvgD"] = dataset["IWD"]
                        dataset["AvgA"] = dataset["IWA"]
                        dataset.drop(columns=["IWH", "IWD", "IWA"], inplace=True)
                    else:
                        dataset["AvgH"] = dataset["WHH"]
                        dataset["AvgD"] = dataset["WHD"]
                        dataset["AvgA"] = dataset["WHA"]
                        dataset.drop(columns=["WHH", "WHD", "WHA"], inplace=True)
                if "WHH" in dataset.columns:
                        dataset.drop(columns=["WHH", "WHD", "WHA"], inplace=True)
                if "SBD" in dataset.columns:
                    dataset.drop(columns=["SBH", "SBD", "SBA"], inplace=True)
                dataset["Country"] = country_name.title()
                dataset["Season"] = season
                dataset["League"] = league["data name"]
                separated_datasets.append(dataset)
    return pd.concat(separated_datasets).set_index(["Date"]).sort_index().reset_index()    