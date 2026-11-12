"""Scrape PSA graded baseball card images and their metadata from www.collectors.com.

USAGE:
    card-grading-scrape --grade 1 --card_num 1 --files_dir ./collection_files --photo_dir ./collection_files/images_psa1

The website regularly goes in and out of service, so failed page loads are retried
after a wait. The scraper runs until the results run out or it is interrupted (ctrl-c);
the metadata is saved after every page and when the scraper stops.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# low and high bound the search on the grade, so only cards of that PSA grade are returned
SEARCH_URL = ("https://www.collectors.com/trading-cards/sport-baseball-cards/20003"
              "?lowgrade={grade}&highgrade={grade}&gradingservice=2&page={page}")
CARD_SELECTOR = "#searchresults > div:nth-child(1) > div > div > div"
ERROR_SELECTORS = ("#error_message", "#error-information-popup-content > div.error-code")

CARDS_PER_PAGE = 24
REQUEST_TIMEOUT_SECONDS = 30
RETRY_WAIT_SECONDS = 5
MAX_RETRY_WAIT_SECONDS = 300
MAX_EMPTY_PAGES = 5  # stop after this many result pages in a row without any cards
MIN_IMAGE_BYTES = 5120  # smaller downloads are placeholder files, not card photos

METADATA_COLUMNS = ["id", "grade", "image_name", "price", "desc", "source"]
UNKNOWN_GRADE = "NA"

_GRADE_PATTERN = re.compile(r"Grade: PSA (\d+(\.\d+)?)")
_EXTENSION_PATTERN = re.compile(r"\.([a-zA-Z]+)$")


@dataclass(frozen=True)
class CardListing:
    """What one search result says about a card. Missing text fields are ``None``."""

    description: str | None
    grade_text: str | None
    price: str | None
    source: str | None
    image_url: str | None
    backup_image_url: str | None

    @property
    def grade_slug(self) -> str | None:
        """The grade as used in file names (``8`` or ``8_5``), or ``None`` if it could not be read."""
        return parse_grade(self.grade_text)


def page_for_card(card_number: int) -> int:
    """The (1-based) results page that the (1-based) card number is on."""
    return (card_number - 1) // CARDS_PER_PAGE + 1


def parse_grade(grade_text: str | None) -> str | None:
    """Turn ``'Grade: PSA 8.5'`` into ``'8_5'``. Returns ``None`` when there is no grade in the text."""
    match = _GRADE_PATTERN.search(grade_text or "")
    return match.group(1).replace(".", "_") if match else None


def file_extension(url: str, default: str = "jpg") -> str:
    """The file extension in the path of the url (ignoring any query string)."""
    match = _EXTENSION_PATTERN.search(urlparse(url).path)
    return match.group(1) if match else default


def _text(card: Tag, name: str, css_class: str) -> str | None:
    element = card.find(name, {"class": css_class})
    return element.text if element is not None else None


def parse_card(card: Tag) -> CardListing:
    image = card.find("img", {"class": "itemimage"})
    return CardListing(
        description=_text(card, "div", "itemdescr"),
        grade_text=_text(card, "div", "itemgrade"),
        # listings without a sale price show a bid price instead
        price=_text(card, "div", "itemsaleprice") or _text(card, "div", "itembidprice"),
        source=_text(card, "a", "itemseller seller-level-med"),
        image_url=image.get("data-zoom-img") if image is not None else None,
        backup_image_url=image.get("backup-img") if image is not None else None,
    )


def parse_results_page(html: bytes) -> list[Tag] | None:
    """The card elements on a results page, or ``None`` if the page is one of the site's error pages."""
    soup = BeautifulSoup(html, "html.parser")
    if any(soup.select(selector) for selector in ERROR_SELECTORS):
        return None
    return soup.select(CARD_SELECTOR)


def fetch(session: requests.Session, url: str) -> requests.Response | None:
    """GET a url. Returns ``None`` if the request failed (the caller is expected to wait and retry)."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        logger.error("request failed for %s: %s", url, error)
        return None


def wait_before_retry(failed_attempts: int) -> None:
    """Sleep with an exponential backoff, capped at ``MAX_RETRY_WAIT_SECONDS``."""
    time.sleep(min(RETRY_WAIT_SECONDS * 2 ** min(failed_attempts - 1, 10), MAX_RETRY_WAIT_SECONDS))


def download_image(session: requests.Session, listing: CardListing) -> tuple[bytes, str] | None:
    """Download the card image, falling back to the backup image. Returns ``(content, url)`` or ``None``."""
    for url in (listing.image_url, listing.backup_image_url):
        if not url:
            continue
        response = fetch(session, url)
        if response is not None and len(response.content) >= MIN_IMAGE_BYTES:
            return response.content, url
    return None


def load_metadata(path: Path) -> pd.DataFrame:
    """Load the metadata of an earlier run, or start an empty table."""
    if path.is_file():
        logger.info("metadata file exists, loading existing metadata")
        return pd.read_pickle(path)
    return pd.DataFrame(columns=METADATA_COLUMNS)


def scrape(grade: int, start_card: int, files_dir: Path, photo_dir: Path,
           session: requests.Session | None = None) -> pd.DataFrame:
    """Scrape the cards of one PSA grade, starting at card number ``start_card``.

    Images are written to ``photo_dir`` as ``card<number>_PSA<grade>.<ext>`` and the
    metadata table is saved to ``files_dir``. Returns the metadata table.
    """
    session = session or requests.Session()
    photo_dir.mkdir(parents=True, exist_ok=True)
    files_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = files_dir / f"PSA{grade}_collectors_metadata.pkl"
    metadata = load_metadata(metadata_path)

    card_number = start_card
    page = page_for_card(card_number)
    # when resuming in the middle of a page, skip the cards on it that come before start_card
    cards_to_skip = (card_number - 1) % CARDS_PER_PAGE
    failed_attempts = 0
    empty_pages = 0

    try:
        while True:
            response = fetch(session, SEARCH_URL.format(grade=grade, page=page))
            cards = parse_results_page(response.content) if response is not None else None
            if cards is None:  # the request failed, or the website is temporarily down
                failed_attempts += 1
                wait_before_retry(failed_attempts)
                continue
            failed_attempts = 0

            # a page without cards is either a broken page or the end of the results
            if not cards:
                empty_pages += 1
                logger.info("page %s: no cards found (%s of %s attempts)", page, empty_pages, MAX_EMPTY_PAGES)
                if empty_pages >= MAX_EMPTY_PAGES:
                    print(f"No more cards found after page {page}, stopping")
                    break
                wait_before_retry(empty_pages)
                continue
            empty_pages = 0

            for card in cards[cards_to_skip:]:
                listing = parse_card(card)
                row = _save_card(session, listing, card_number, page, photo_dir)
                if row is not None:
                    metadata.loc[card_number] = row
                card_number += 1
            cards_to_skip = 0

            metadata.to_pickle(metadata_path)
            # a page can hold fewer than 24 cards, so the next page is not derived from the card number
            page += 1
    except KeyboardInterrupt:
        print("Script interrupted by user")
    finally:
        metadata.to_pickle(metadata_path)
    return metadata


def _save_card(session: requests.Session, listing: CardListing, card_number: int, page: int,
               photo_dir: Path) -> list | None:
    """Download the image of one card. Returns its metadata row, or ``None`` if the card was skipped."""
    # the grade is part of the file name (and later the training label), so it is never guessed
    if listing.grade_slug is None:
        logger.error("page %s: card %s: grade unknown, skipping card", page, card_number)
        return None
    image = download_image(session, listing)
    if image is None:
        logger.error("page %s: card %s: image download failed, skipping card", page, card_number)
        return None
    content, url = image

    image_path = photo_dir / f"card{card_number}_PSA{listing.grade_slug}.{file_extension(url)}"
    image_path.write_bytes(content)

    def or_nan(value):
        return np.nan if value is None else value

    return [card_number, listing.grade_text or UNKNOWN_GRADE, str(image_path),
            or_nan(listing.price), or_nan(listing.description), or_nan(listing.source)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape PSA1-10 images from www.collectors.com")
    parser.add_argument("-g", "--grade", type=int, choices=range(1, 11), required=True,
                        help="PSA grade value (1-10) of cards to scrape")
    parser.add_argument("-c", "--card_num", type=int, default=1,
                        help="Card number to start scraping from, there are 24 cards per page")
    parser.add_argument("-f", "--files_dir", type=Path, required=True,
                        help="Path to dir for metadata and log data saves")
    parser.add_argument("-p", "--photo_dir", type=Path, required=True,
                        help="Path to dir for photo download")
    args = parser.parse_args(argv)

    args.files_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.files_dir / f"PSA{args.grade}_collectors_LOGS.txt",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")
    scrape(args.grade, args.card_num, args.files_dir, args.photo_dir)


if __name__ == "__main__":
    main()
