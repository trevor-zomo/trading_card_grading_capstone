import csv
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("cv2")

from card_grading import capture  # noqa: E402

BACKGROUND = (25, 25, 28)
BORDER = (245, 245, 238)


def make_card(color, size=(750, 1050), border=40):
    card = Image.new("RGB", size, BORDER)
    card.paste(color, (border, border, size[0] - border, size[1] - border))
    return card


def on_scanner_background(card, margin=80, angle=3):
    """The card as a feeder scans it when nothing crops it: slightly crooked, with background around it."""
    rotated = card.rotate(angle, expand=True, fillcolor=BACKGROUND, resample=Image.BICUBIC)
    scan = Image.new("RGB", (rotated.width + 2 * margin, rotated.height + 2 * margin), BACKGROUND)
    scan.paste(rotated, (margin, margin))
    return scan


def save(image, path, dpi=300):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(dpi, dpi))


def read_catalogue(library):
    with open(library / capture.CATALOGUE_FILE, newline="") as f:
        return list(csv.DictReader(f))


def names(card_scans):
    return [(scan.front.name, scan.back.name if scan.back else None) for scan in card_scans]


def test_fastfoto_files_are_paired_by_their_suffix():
    files = ["box_0002.jpg", "box_0010.jpg", "box_0001_b.jpg", "box_0001.jpg", "box_0001_a.jpg", "box_0002_b.jpg"]
    assert names(capture.pair_scans([Path(name) for name in files])) == [
        ("box_0001.jpg", "box_0001_b.jpg"), ("box_0002.jpg", "box_0002_b.jpg"), ("box_0010.jpg", None)]


def test_alternating_pages_are_paired_in_page_order():
    files = ["page_10.png", "page_2.png", "page_1.png", "page_11.png", "page_12.png"]
    paths = [Path(name) for name in files]
    assert names(capture.pair_scans(paths)) == [
        ("page_1.png", "page_2.png"), ("page_10.png", "page_11.png"), ("page_12.png", None)]
    assert names(capture.pair_scans(paths, single_sided=True))[1] == ("page_2.png", None)


def test_image_of_exactly_one_card_is_only_turned_upright():
    upright = make_card((200, 40, 40))
    upright.info["dpi"] = (300, 300)
    assert capture.isolate_card(upright)[0].size == (750, 1050)

    sideways = upright.rotate(90, expand=True)
    sideways.info["dpi"] = (300, 300)
    card, note = capture.isolate_card(sideways)
    assert (card.size, note) == ((750, 1050), "")

    without_resolution = make_card((200, 40, 40), size=(500, 700))
    assert capture.isolate_card(without_resolution) == (without_resolution, "")


def test_card_with_scanner_background_around_it_is_cropped_and_straightened():
    scan = on_scanner_background(make_card((40, 160, 60)))
    scan.info["dpi"] = (300, 300)
    card, note = capture.isolate_card(scan)
    assert note == ""
    assert abs(card.width - 750) <= 8 and abs(card.height - 1050) <= 8
    assert max(abs(a - b) for a, b in zip(card.getpixel((375, 12)), BORDER, strict=True)) < 40


def test_image_without_a_recognisable_card_is_kept_and_flagged():
    blank = Image.new("RGB", (1700, 2200), BACKGROUND)
    blank.info["dpi"] = (300, 300)
    card, note = capture.isolate_card(blank)
    assert card.size == (1700, 2200)
    assert "no card" in note


def test_folder_from_scanner_software_is_imported_and_a_batch_can_be_added_to(tmp_path):
    folder, library = tmp_path / "fastfoto", tmp_path / "library"
    save(make_card((200, 40, 40)), folder / "box_0001.jpg")
    save(make_card((60, 60, 60)), folder / "box_0001_b.jpg")
    save(make_card((200, 40, 40)), folder / "box_0001_a.jpg")                      # enhanced copy: not needed
    save(on_scanner_background(make_card((40, 160, 60))), folder / "box_0002.jpg")
    save(make_card((60, 60, 60)).rotate(90, expand=True), folder / "box_0002_b.jpg")
    save(make_card((50, 80, 200)), folder / "box_0003.jpg")                         # its back is missing
    (folder / "notes.txt").write_text("not a scan")

    capture.main(["--library", str(library), "--batch", "box01", "--from_folder", str(folder)])

    rows = read_catalogue(library)
    assert [row["card_id"] for row in rows] == ["box01-0001", "box01-0002", "box01-0003"]
    assert [row["status"] for row in rows] == ["ok", "ok", "needs_review"]
    assert rows[2]["note"] == "no back image" and rows[2]["back_file"] == ""
    assert rows[0]["front_file"] == "images/box01/box01-0001_front.jpg" and rows[0]["dpi"] == "300"
    for row in rows[:2]:
        for side in ("front_file", "back_file"):
            with Image.open(library / row[side]) as image:
                assert abs(image.width - 750) <= 8 and abs(image.height - 1050) <= 8   # cropped and upright
                assert round(image.info["dpi"][0]) == 300

    # the next feeder load of the same box continues the numbering
    capture.main(["--library", str(library), "--batch", "box01", "--from_folder", str(folder)])
    assert [row["card_id"] for row in read_catalogue(library)][3:] == ["box01-0004", "box01-0005", "box01-0006"]


def fake_scanimage(pages, stderr="scanimage: sane_start: Document feeder out of documents"):
    """Stands in for scanimage: writes the given pages to the --batch pattern and records the command."""
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        pattern = next(option for option in command if option.startswith("--batch=")).split("=", 1)[1]
        for number, page in enumerate(pages, start=1):
            page.save(pattern % number, dpi=(600, 600))

        class Result:
            returncode = 7
        Result.stderr = stderr
        return Result()

    return run, calls


def test_scanning_the_feeder_fills_the_library(tmp_path, monkeypatch):
    front, back = make_card((200, 40, 40), size=(1500, 2100), border=80), make_card((60, 60, 60), size=(1500, 2100))
    run, calls = fake_scanimage([front, back, front, back])
    monkeypatch.setattr(capture.subprocess, "run", run)

    capture.main(["--library", str(tmp_path), "--batch", "box02", "--device", "epsonds:libusb:001:004",
                  "--scanimage_option=--depth=8"])

    (command,) = calls
    assert command[:3] == ["scanimage", "--device-name", "epsonds:libusb:001:004"]
    assert command[command.index("--source") + 1] == "ADF Duplex"
    assert command[command.index("--resolution") + 1] == "600"
    assert {"--adf-crp=yes", "--adf-skew=yes", "--format=png", "--depth=8"} <= set(command)
    rows = read_catalogue(tmp_path)
    assert [(row["card_id"], row["status"], row["dpi"]) for row in rows] == [
        ("box02-0001", "ok", "600"), ("box02-0002", "ok", "600")]
    assert sorted(path.name for path in (tmp_path / "images" / "box02").iterdir()) == [
        "box02-0001_back.jpg", "box02-0001_front.jpg", "box02-0002_back.jpg", "box02-0002_front.jpg"]


def test_fronts_only_and_no_driver_crop(tmp_path, monkeypatch):
    run, calls = fake_scanimage([make_card((200, 40, 40), size=(1500, 2100))] * 3)
    monkeypatch.setattr(capture.subprocess, "run", run)
    rows = capture.scan_into_library(tmp_path, "box03", single_sided=True, driver_crop=False)
    assert len(rows) == 3 and all(row["note"] == "no back image" for row in rows)
    assert calls[0][calls[0].index("--source") + 1] == "ADF Front"
    assert "--adf-crp=yes" not in calls[0]


def test_problems_are_reported_plainly(tmp_path, monkeypatch):
    def missing(command, **kwargs):
        raise FileNotFoundError("scanimage")

    monkeypatch.setattr(capture.subprocess, "run", missing)
    with pytest.raises(SystemExit, match="--from_folder"):
        capture.main(["--library", str(tmp_path), "--batch", "box01"])

    run, _ = fake_scanimage([], stderr="scanimage: no SANE devices found")
    monkeypatch.setattr(capture.subprocess, "run", run)
    with pytest.raises(SystemExit, match="no SANE devices found"):
        capture.main(["--library", str(tmp_path), "--batch", "box01"])

    with pytest.raises(SystemExit, match="600 dpi"):
        capture.main(["--library", str(tmp_path), "--batch", "box01", "--resolution", "1200"])
    with pytest.raises(SystemExit, match="batch name"):
        capture.main(["--library", str(tmp_path), "--batch", "../elsewhere", "--from_folder", str(tmp_path)])
    assert not (tmp_path / capture.CATALOGUE_FILE).exists()


def test_batch_name_is_checked_before_the_scanner_runs(tmp_path, monkeypatch):
    run, calls = fake_scanimage([make_card((200, 40, 40))])
    monkeypatch.setattr(capture.subprocess, "run", run)
    with pytest.raises(SystemExit, match="batch name"):
        capture.main(["--library", str(tmp_path), "--batch", "my box"])
    assert calls == []
