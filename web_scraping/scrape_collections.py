import os
import re
import sys
import time
import requests
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup

"""
Insert Script Description Here..

USAGE:
python scrape_collections.py --grade {} --card_num {} --files_dir {} --photo_dir {}

ARGS:
--grade     (-g): (INT): The PSA grade value (1-10) of cards to scrape from www.collectors.com
--card_num  (-c): (INT): The card number to start scraping from, there are 24 cards per page
--files_dir (-f): (STR): The path to where you want the metadata and scripting log files to be saved
--photo_dir (-p): (STR): The path to where you want the photos downloaded

EXAMPLE:
python scrape_collections.py
--grade 1 
--card_num 1
--files_dir /Users/brianmiller/Desktop/collection_files 
--photo_dir /Users/brianmiller/Desktop/collection_files/images_psa1 
"""

# define and parse the user args 
parser = argparse.ArgumentParser(description='Scrape PSA1-10 images from www.collectors.com')
parser.add_argument('-g', '--grade', type=int, choices=range(1, 11), required=True, help='PSA grade value (1-10) of cards to scrape')
parser.add_argument('-c', '--card_num', type=int, default=1, help='Card number to start scraping from')
parser.add_argument('-f', '--files_dir', type=str, required=True, help='Path to dir for metadata and log data saves')
parser.add_argument('-p', '--photo_dir', type=str, required=True, help='Path to dir for photo download')
args = parser.parse_args()


def load_metadata(metadata_pickle_file_location):
    """_summary_

    Args:
        metadata_pickle_file_location (str):
    Returns:
        pandas.dataframe: _description_
    """

    # if the metadata file exists, load it
    if os.path.isfile(metadata_pickle_file_location):
        print("LOG: Picklefile exists, loading existing metadata")
        metadata_df = pd.read_pickle(metadata_pickle_file_location)
    
    # if it does not exist, make it
    else:
        # Create an empty DataFrame, save it, and reopen it, this helps if the script fails without warning
        metadata_df = pd.DataFrame(columns=['id', 'grade', 'image_name', 'price', 'desc', 'source'])
        metadata_df.to_pickle(metadata_pickle_file_location)  # save the empty metadata
        metadata_df = pd.read_pickle(metadata_pickle_file_location)
        
    return metadata_df


def calc_page_counter_from_card_counter(card_counter):
    """
    Given the card counter, calculate what page this scraping script should jump to in order to skip already scraped images
    Args:
        card_counter (int): counter for card ids
    Returns:
        the page counter int of what page the script should start on
    """
    return ((card_counter - 1) // CARDS_PER_PAGE) + 1


def polite_get(url):
    """
    GET a url with a timeout. Returns the response, or None if the request failed
    (the caller is expected to wait and retry).
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        log_file.write("ERROR: request failed for {}: {}\n".format(url, e))
        return None


def wait_before_retry(failed_attempts):
    """Sleep with an exponential backoff, capped at MAX_RETRY_WAIT_SECONDS."""
    time.sleep(min(RETRY_WAIT_SECONDS * (2 ** min(failed_attempts - 1, 10)), MAX_RETRY_WAIT_SECONDS))


CARDS_PER_PAGE = 24
REQUEST_TIMEOUT_SECONDS = 30
RETRY_WAIT_SECONDS = 5
MAX_RETRY_WAIT_SECONDS = 300
MAX_EMPTY_PAGES = 5  # stop after this many result pages in a row without any cards

# define the output locations
image_save_dir = args.photo_dir
metadata_pickle_filename = "{}/PSA{}_collectors_metadata.pkl".format(args.files_dir, args.grade)
log_filename = "{}/PSA{}_collectors_LOGS.txt".format(args.files_dir, args.grade)

# define the base search url, low and high bound the search on their website so we only get PSA grades of that int
base_url = "https://www.collectors.com/trading-cards/sport-baseball-cards/20003?lowgrade={}&highgrade={}&gradingservice=2&page=".format(args.grade, args.grade)

# load the metadata pandas df, if it doesn't exist, make one
metadata_df = load_metadata(metadata_pickle_filename)

# open a log file, if it doesn't exist, make one
log_file = open(log_filename, 'a')

# initiate a counter of what card to start on
card_counter = args.card_num

# when resuming in the middle of a page, skip the cards on that page that come before card_counter
cards_to_skip = (card_counter - 1) % CARDS_PER_PAGE

# the page of results to load first, after that the pages are simply loaded one after the other
page_counter = calc_page_counter_from_card_counter(card_counter)

# counters for backing off when the website is down, and for noticing the end of the results
failed_attempts = 0
empty_pages = 0

# until ctrl-c to exit (or the results run out), do the following
# the website that I am scraping constantly goes in and out of service,
# so failed page loads are retried after a wait, to 'wait out' those times when the page won't load
try:
    while True:
        page_url = base_url + str(page_counter)

        # GET the page, parse the response using Beautiful Soup
        response = polite_get(page_url)
        if response is None:
            failed_attempts += 1
            wait_before_retry(failed_attempts)
            continue
        soup = BeautifulSoup(response.content, "html.parser")

        # check to see if the website is live, sometimes it is simply broken
        if soup.select("#error_message"):
            # print("ERROR: WEBSITE DOWN TEMPORARILY: Internal error: Page {}\n".format(page_counter))
            # log_file.write("ERROR: WEBSITE DOWN TEMPORARILY: Internal error: Page {}\n".format(page_counter))
            failed_attempts += 1
            wait_before_retry(failed_attempts)
            continue
            
        elif soup.select("#error-information-popup-content > div.error-code"):
            # print("ERROR: WEBSITE DOWN TEMPORARILY: 503 error: Page {}\n".format(page_counter))
            # log_file.write("ERROR: WEBSITE DOWN TEMPORARILY: 503 error: Page {}\n".format(page_counter))
            failed_attempts += 1
            wait_before_retry(failed_attempts)
            continue

        # use CSS selector, to get all of the 24 divs, for the 24 card results, per page. recursive = False to not get the sub divs
        # (the '>' child combinators already keep the sub divs out)
        child_divs = soup.select("#searchresults > div:nth-child(1) > div > div > div")

        # a page without cards is either a broken page or the end of the results, retry a few times then stop
        if not child_divs:
            empty_pages += 1
            log_file.write("LOG: page {}: no cards found ({} of {} attempts)\n".format(page_counter, empty_pages, MAX_EMPTY_PAGES))
            if empty_pages >= MAX_EMPTY_PAGES:
                print("No more cards found after page {}, stopping".format(page_counter))
                break
            wait_before_retry(empty_pages)
            continue
        failed_attempts = 0
        empty_pages = 0

        # skip the cards that were already scraped (only applies to the first page after a restart)
        child_divs = child_divs[cards_to_skip:]
        cards_to_skip = 0

        # for each of the cards to scrape
        for child_div in child_divs:

            # CARD DESCRIPTION
            try:
                desc_div = child_div.find("div", {"class": "itemdescr"})
                desc_txt = desc_div.text

            except AttributeError:
                log_file.write("LOG: page {}: card {}: Description: AttributeError: returning NA\n".format(page_counter, card_counter))
                desc_txt = np.nan


            # CARD GRADE
            grade_num_str = None  # reset, so a failed lookup can't reuse the grade of the previous card
            try:
                grade_div = child_div.find("div", {"class": "itemgrade"})
                grade_txt = grade_div.text

                # get the grade int from the string
                pattern = r"Grade: PSA (\d+(\.\d+)?)"  # matches one or more digits followed by an optional decimal point and one or more digits
                match = re.search(pattern, grade_txt)  # find the first instance of this pattern
                if match:
                    # Get the matched number and replace "." with "_"
                    grade_num_str = match.group(1).replace(".", "_")
                else:
                    log_file.write("ERROR: page {}: card {}: regex failed: returning 'NA' \n".format(page_counter, card_counter))
                    grade_txt = "NA"

            except AttributeError:
                log_file.write("ERROR: page {}: card {}: Grade: AttributeError: returning 'NA' \n".format(page_counter, card_counter))
                grade_txt = "NA"


            # CARD PRICE
            try:
                price_div = child_div.find("div", {"class": "itemsaleprice"})
                price_txt = price_div.text

            except AttributeError:
                log_file.write("LOG: page {}: card {}: Price: AttributeError: Price data not found, trying Bid data\n".format(page_counter, card_counter))
                try:
                    price_div = child_div.find("div", {"class": "itembidprice"})
                    price_txt = price_div.text
                except AttributeError:
                    log_file.write("LOG: page {}: card {}: Bid: AttributeError: Bid data not found, returning NA\n".format(page_counter, card_counter))
                    price_txt = np.nan



            # LISTING SOURCE (e.g. EBAY, AMAZON, etc.)
            try:
                source_a = child_div.find("a", {"class": "itemseller seller-level-med"})
                source_txt = source_a.text

            except AttributeError:
                log_file.write("LOG: page {}: card {}: Source: AttributeError: returning NA\n".format(page_counter, card_counter))
                source_txt = np.nan

            ## IMAGE DOWNLOAD
            try:
                # grab the image element
                img_element = child_div.find("img", {"class": "itemimage"})

                # the grade is part of the file name (and later the training label), so don't guess it
                if grade_num_str is None:
                    log_file.write("ERROR: page {}: card {}: Image: grade unknown, skipping image\n".format(page_counter, card_counter))
                    card_counter +=1
                    continue  # skip to the next image

                # image URL attempt 1 (works in most cases)
                image_url = img_element["data-zoom-img"]

                # image URL attempt 2
                # if the resulting image is less than 5 kb, it's probably not an actual image,
                # just a placeholder file in their database, so try option 2
                image_response = polite_get(image_url)  # load image1 into memory
                if image_response is None or len(image_response.content) < 5120:  # 5kb = 5120 bytes
                    log_file.write("LOG: page {}: card {}: Image: Image < 5KB, attempting image 2\n".format(page_counter, card_counter))
                    image_url = img_element["backup-img"]  # still not fully confident in this, but it does work.
                    image_response = polite_get(image_url) # load image2 into memory, replacing image1

                if image_response is None:
                    log_file.write("ERROR: page {}: card {}: Image: download failed\n".format(page_counter, card_counter))
                    card_counter +=1
                    continue  # skip to the next image

                # creating the card image filename
                # get the file extension from the path of the url (ignoring any query string), default to jpg
                match = re.search(r'\.([a-zA-Z]+)$', urlparse(image_url).path)
                file_extension = match.group(1) if match else "jpg"
                cardimage_filename = image_save_dir + "/card" + str(card_counter) + "_PSA" + grade_num_str + "." + file_extension  # full image path

                # saving ths image
                with open(cardimage_filename, "wb") as f:
                    f.write(image_response.content)  # save it

            except AttributeError:
                log_file.write("ERROR: page {}: card {}: Image: AttributeError\n".format(page_counter, card_counter))
                card_counter +=1
                continue  # skip to the next image
            except KeyError:
                log_file.write("ERROR: page {}: card {}: Image: KeyError (data-zoom-img not found)\n".format(page_counter, card_counter))
                card_counter +=1
                continue  # skip to the next image

            # print(card_counter)
            # print(desc_txt)
            # print(grade_txt)
            # print(grade_num_str)
            # print(price_txt)
            # print(source_txt)
            # print(cardimage_filename)
            # print(image_url)
            # print("\n")

            # add metadata for card to the metadata df
            metadata_df.loc[card_counter] = [card_counter, grade_txt, cardimage_filename, price_txt, desc_txt, source_txt]
            metadata_df.to_pickle(metadata_pickle_filename)  # save the metadata

            # go to the next card
            card_counter +=1

        # go to the next page (a page can hold fewer than 24 cards, so don't derive it from the card counter)
        page_counter += 1
            
except KeyboardInterrupt:
    # Handle the case when the user presses Ctrl+C
    print('Script interrupted by user')
finally:
    # however the script ends, save the metadata and close the log file
    metadata_df.to_pickle(metadata_pickle_filename)
    log_file.close()

sys.exit(0)
