import pytest

pytest.importorskip("bs4")
pytest.importorskip("pandas")
requests = pytest.importorskip("requests")

from card_grading import scraping  # noqa: E402


def card_html(grade="Grade: PSA 8", image="http://img.test/a.jpg?v=1"):
    return (f'<div><div class="itemdescr">1989 Topps</div><div class="itemgrade">{grade}</div>'
            f'<div class="itemsaleprice">$5.00</div><a class="itemseller seller-level-med">eBay (1)</a>'
            f'<img class="itemimage" data-zoom-img="{image}"/></div>')


def results_page(*cards):
    return f'<div id="searchresults"><div><div><div>{"".join(cards)}</div></div></div></div>'.encode()


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    """Serves result pages from a dict, and 6 kB of bytes for any image url."""

    def __init__(self, pages, failures=0):
        self.pages = pages
        self.failures = failures
        self.page_requests = []

    def get(self, url, timeout):
        assert timeout
        if "page=" not in url:
            return FakeResponse(b"x" * 6000)
        if self.failures:
            self.failures -= 1
            raise requests.ConnectionError("down")
        page = int(url.split("page=")[1])
        self.page_requests.append(page)
        return FakeResponse(self.pages.get(page, results_page()))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(scraping.time, "sleep", lambda seconds: None)


@pytest.mark.parametrize("card, page", [(1, 1), (5, 1), (24, 1), (25, 2), (48, 2), (49, 3), (3104, 130)])
def test_page_for_card(card, page):
    assert scraping.page_for_card(card) == page


@pytest.mark.parametrize("text, slug", [
    ("Grade: PSA 8", "8"), ("Grade: PSA 8.5", "8_5"), ("Grade: PSA 10", "10"), ("no grade", None), (None, None),
])
def test_parse_grade(text, slug):
    assert scraping.parse_grade(text) == slug


@pytest.mark.parametrize("url, extension", [
    ("http://x/a.jpg", "jpg"), ("http://x/a.png?v=2", "png"), ("http://x/image", "jpg"),
])
def test_file_extension(url, extension):
    assert scraping.file_extension(url) == extension


def test_parse_card_reads_fields_and_falls_back_to_bid_price():
    (card,) = scraping.parse_results_page(results_page(
        card_html().replace("itemsaleprice", "itembidprice")))
    listing = scraping.parse_card(card)
    assert listing.description == "1989 Topps"
    assert listing.grade_slug == "8"
    assert listing.price == "$5.00"
    assert listing.source == "eBay (1)"
    assert listing.image_url == "http://img.test/a.jpg?v=1"


def test_error_page_is_not_a_results_page():
    assert scraping.parse_results_page(b'<div id="error_message">oops</div>') is None


def test_scrape_names_files_by_grade_skips_unknown_grades_and_stops(tmp_path):
    session = FakeSession({1: results_page(
        card_html(), card_html(grade="unknown"), card_html(grade="Grade: PSA 8.5", image="http://img.test/c.png"))},
        failures=1)

    metadata = scraping.scrape(8, 1, tmp_path, tmp_path / "img", session=session)

    assert sorted(path.name for path in (tmp_path / "img").iterdir()) == ["card1_PSA8.jpg", "card3_PSA8_5.png"]
    assert list(metadata["id"]) == [1, 3]
    assert list(metadata["grade"]) == ["Grade: PSA 8", "Grade: PSA 8.5"]
    # page 1 once, then page 2 until the scraper gives up on finding more cards
    assert session.page_requests == [1] + [2] * scraping.MAX_EMPTY_PAGES
    assert (tmp_path / "PSA8_collectors_metadata.pkl").is_file()


def test_scrape_resumes_in_the_middle_of_a_page(tmp_path):
    session = FakeSession({1: results_page(card_html(), card_html(), card_html())})
    metadata = scraping.scrape(8, 3, tmp_path, tmp_path / "img", session=session)
    assert list(metadata["id"]) == [3]
