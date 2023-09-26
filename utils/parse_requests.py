import sys
import requests
from bs4 import BeautifulSoup as bsoup
import time

def request_download(url):
    """Request to get the URL passed and return the response."""
    
    max_retries = 5  # Maximum number of retries before raising an exception
    wait_time = 10  # Initial wait time in seconds
    
    for retry_count in range(max_retries + 1):
        try:
            res = requests.get(url)
            res.raise_for_status()
        except (requests.exceptions.RequestException) as err:
            print(f"Couldn't Get Request: {err}", file=sys.stderr)
            
            if retry_count == max_retries:
                raise Exception(f"Process Terminated: {err}")
            
            for i in range(reversed(wait_time)):
                print(f"Retrying in {i} seconds...", end="\r")
                time.sleep(wait_time)
            wait_time += 3
        else:
            break
    
    return res

            
def request_and_parse(url):
    """Request and parse the passed url, Returns the parsed response."""
    res = request_download(url)
    print("Downloaded, Now Parsing")
    soup = bsoup(res.text, features="html5lib")
    return soup
            
