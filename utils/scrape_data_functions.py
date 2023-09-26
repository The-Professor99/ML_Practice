import os
import time
from urllib.parse import urljoin
from utils.parse_requests import request_download, request_and_parse
from utils.file_manipulation import save_data

def scrape_football_data(main_folder_path: str, update_only: bool=False, season: str="2324") -> None:
    """
    Scraps football data from https://www.football-data.co.uk
    
    Params
    ------
    main_folder_path: folder path of where the files will be downloaded to.
    update_only: if True, only the data for the specified `season` is downloaded. some categories, eg usa, aren't grouped into seasons. These categories have one csv file containing all the seasons grouped together. if update_only, these files are downloaded too.
    season: the season to download data for. Only used if update_only is specified.

    """
    url = "https://www.football-data.co.uk" 
    soup = request_and_parse(urljoin(url, "data.php"))
    tags_to_scrap = soup.select("td > a")
    for link in tags_to_scrap:
        result_name = link.get_text()
        # Check if "Results" is not in the result_name
        if "Results" not in result_name:
            continue

        country_name = result_name.lower().split(" ")[0]
        result_link = urljoin(url, link.get("href"))
        soup = request_and_parse(result_link)
        time.sleep(2) # sleep for 2 seconds so we don't send too many requests at a time to host

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
            if update_only and yr != season and yr != "all":
                continue
 
            # Request and download the CSV data
            res = request_download(urljoin(url, datalink))

            # Save the data to the specified folder
            save_data(res, filename, os.path.join(main_folder_path, country_name, yr))
            time.sleep(2)
            print(f"Downloaded: {country_name} - {yr} - {filename}")
