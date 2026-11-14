"""Scan cards with a sheet-fed duplex scanner, or import its scans, into a card library.

A library is a folder that holds one image of the front and one of the back of every card,
and a catalogue with one row per card:

    <library>/cards.csv
    <library>/images/<batch>/<batch>-0001_front.jpg
    <library>/images/<batch>/<batch>-0001_back.jpg

USAGE:
    # scan the cards in the feeder (Linux and macOS, needs SANE's scanimage; the Epson
    # FastFoto FF-680W is supported over USB)
    card-grading-scan --library ./my_cards --batch box01

    # or import what the scanner's own software saved to a folder
    card-grading-scan --library ./my_cards --batch box01 --from_folder ./fastfoto_output

Running the command again with the same batch name adds to it, so a box can be scanned one
feeder load at a time. Every image is cropped to the card and turned upright where the card can
be told apart from its surroundings; cards that need a look are marked in the catalogue.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from card_grading import scans

CATALOGUE_FILE = "cards.csv"
CATALOGUE_COLUMNS = ["card_id", "batch", "position", "front_file", "back_file", "dpi", "status", "note", "scanned_at"]
STATUS_OK, STATUS_REVIEW = "ok", "needs_review"

DEFAULT_RESOLUTION = 600
# the scanner's document feeder does not go above this resolution
MAX_FEEDER_RESOLUTION = 600

# Epson FastFoto saves the back of a photo with this suffix, and an enhanced copy of the front with the other
BACK_SUFFIX, ENHANCED_SUFFIX = "_b", "_a"

# How far an image may be from the real card size (when it states its resolution) or from the
# proportions of a card (when it does not) and still count as showing exactly one card
CARD_SIZE_TOLERANCE = 0.05
CARD_ASPECT_TOLERANCE = 0.04


class CaptureError(Exception):
    """The scan or import could not be carried out."""


@dataclass(frozen=True)
class CardScan:
    """The scans of one card, as files that still have to be imported."""

    front: Path
    back: Path | None


def _natural_key(path: Path) -> list:
    """Sort key that puts page2 before page10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def pair_scans(paths: list[Path], single_sided: bool = False) -> list[CardScan]:
    """Work out which files are the front and the back of the same card.

    Two layouts are understood: Epson FastFoto's, where the back of ``name.jpg`` is ``name_b.jpg``
    (and ``name_a.jpg`` is an enhanced copy that is not needed), and a plain sequence of pages
    in which fronts and backs alternate, as a duplex scan produces.
    """
    by_stem = {path.stem: path for path in paths}
    backs = {stem for stem in by_stem if stem.endswith(BACK_SUFFIX) and stem[:-len(BACK_SUFFIX)] in by_stem}
    enhanced = {stem for stem in by_stem
                if stem.endswith(ENHANCED_SUFFIX) and stem[:-len(ENHANCED_SUFFIX)] in by_stem}
    if backs or enhanced:
        fronts = sorted((path for stem, path in by_stem.items() if stem not in backs | enhanced), key=_natural_key)
        return [CardScan(front, by_stem.get(front.stem + BACK_SUFFIX)) for front in fronts]

    ordered = sorted(paths, key=_natural_key)
    if single_sided:
        return [CardScan(front, None) for front in ordered]
    return [CardScan(ordered[i], ordered[i + 1] if i + 1 < len(ordered) else None)
            for i in range(0, len(ordered), 2)]


def _is_one_card(width: int, height: int, dpi: float | None) -> bool:
    """Whether an image of this size shows exactly one card, edge to edge."""
    short, long = sorted((width, height))
    if dpi:
        return (abs(short / (scans.CARD_SHORT_INCHES * dpi) - 1) <= CARD_SIZE_TOLERANCE
                and abs(long / (scans.CARD_LONG_INCHES * dpi) - 1) <= CARD_SIZE_TOLERANCE)
    return abs(short / long - scans.CARD_ASPECT) <= CARD_ASPECT_TOLERANCE


def isolate_card(image: Image.Image) -> tuple[Image.Image, str]:
    """Crop an image to the card it shows and turn it upright.

    Returns the image and a note. The note is empty when the result shows exactly one card;
    otherwise the image is returned whole and the note says what needs a look.
    """
    dpi = scans.stated_dpi(image)
    image = image.convert("RGB")
    if _is_one_card(image.width, image.height, dpi):
        return (image.rotate(-90, expand=True) if image.width > image.height else image), ""

    scan = np.asarray(image)
    rectangles, _ = scans.find_cards(scan, dpi, size_tolerance=CARD_SIZE_TOLERANCE)
    if len(rectangles) == 1:
        return Image.fromarray(scans.extract_card(scan, rectangles[0])), ""
    if not rectangles:
        return image, "no card could be told apart from the background"
    return image, f"{len(rectangles)} cards in one image"


def scan_pages(folder: Path, resolution: int = DEFAULT_RESOLUTION, device: str | None = None,
               duplex: bool = True, driver_crop: bool = True, extra_options: tuple[str, ...] = ()) -> list[Path]:
    """Scan everything in the scanner's feeder into ``folder`` with SANE's ``scanimage``. Returns the pages.

    With ``duplex`` the pages alternate between fronts and backs.
    """
    if resolution > MAX_FEEDER_RESOLUTION:
        raise CaptureError(f"The document feeder scans at {MAX_FEEDER_RESOLUTION} dpi at most")
    command = ["scanimage", "--source", "ADF Duplex" if duplex else "ADF Front", "--mode", "Color",
               "--resolution", str(resolution), "--format=png", f"--batch={folder / 'page_%04d.png'}"]
    if device:
        command[1:1] = ["--device-name", device]
    if driver_crop:
        # let the scanner driver straighten each card and crop to it
        command += ["--adf-skew=yes", "--adf-crp=yes"]
    command += list(extra_options)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise CaptureError(
            "scanimage was not found. Direct scanning needs SANE (the sane-backends package on Linux and macOS). "
            "Otherwise scan with the scanner's own software and import its folder with --from_folder.") from error
    pages = sorted(folder.glob("page_*.png"), key=_natural_key)
    if not pages:
        # scanimage also ends with an error when the feeder runs empty, so only this case is a failure
        raise CaptureError(f"The scanner returned no pages. scanimage said: {result.stderr.strip() or 'nothing'}")
    return pages


def _read_catalogue(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_batch_name(batch: str) -> None:
    """Batch names become folder and file names, so they are kept simple."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", batch):
        raise CaptureError("A batch name may only hold letters, digits, dots, dashes and underscores")


def import_cards(card_scans: list[CardScan], library: Path, batch: str) -> list[dict]:
    """Add scanned cards to a library. Returns the catalogue rows that were added."""
    check_batch_name(batch)
    catalogue_path = library / CATALOGUE_FILE
    existing = _read_catalogue(catalogue_path)
    position = max((int(row["position"]) for row in existing if row["batch"] == batch), default=0)
    image_dir = library / "images" / batch
    image_dir.mkdir(parents=True, exist_ok=True)
    scanned_at = datetime.datetime.now().isoformat(timespec="seconds")

    rows = []
    for card_scan in card_scans:
        position += 1
        card_id = f"{batch}-{position:04d}"
        row = {"card_id": card_id, "batch": batch, "position": position, "scanned_at": scanned_at}
        notes = []
        for side, source in (("front", card_scan.front), ("back", card_scan.back)):
            if source is None:
                notes.append(f"no {side} image")
                continue
            with Image.open(source) as image:
                dpi = scans.stated_dpi(image)
                card, note = isolate_card(image)
            file_name = f"{card_id}_{side}.jpg"
            card.save(image_dir / file_name, quality=95, **({"dpi": (dpi, dpi)} if dpi else {}))
            row[f"{side}_file"] = f"images/{batch}/{file_name}"
            row["dpi"] = round(dpi) if dpi else ""
            if note:
                notes.append(f"{side}: {note}")
        row["status"] = STATUS_REVIEW if notes else STATUS_OK
        row["note"] = "; ".join(notes)
        rows.append(row)

    with open(catalogue_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOGUE_COLUMNS, restval="")
        if not existing:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def import_folder(folder: Path, library: Path, batch: str, single_sided: bool = False) -> list[dict]:
    """Import the scans that scanner software saved to a folder."""
    check_batch_name(batch)
    paths = [path for path in folder.iterdir() if path.suffix.lower() in scans.SCAN_SUFFIXES]
    if not paths:
        raise CaptureError(f"No scans found in {folder}")
    return import_cards(pair_scans(paths, single_sided), library, batch)


def scan_into_library(library: Path, batch: str, resolution: int = DEFAULT_RESOLUTION, device: str | None = None,
                      single_sided: bool = False, driver_crop: bool = True,
                      extra_options: tuple[str, ...] = ()) -> list[dict]:
    """Scan the cards in the feeder and add them to a library."""
    check_batch_name(batch)   # before the scan, so that a typo does not cost a feeder load
    with tempfile.TemporaryDirectory() as temporary:
        pages = scan_pages(Path(temporary), resolution, device, duplex=not single_sided,
                           driver_crop=driver_crop, extra_options=extra_options)
        return import_cards(pair_scans(pages, single_sided), library, batch)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan cards, or import scans, into a card library")
    parser.add_argument("-l", "--library", type=Path, required=True, help="Folder of the card library")
    parser.add_argument("-b", "--batch", required=True,
                        help="Name for this group of cards, e.g. the box they are in; an existing batch is added to")
    parser.add_argument("--from_folder", type=Path, default=None,
                        help="Import the scans in this folder instead of driving the scanner")
    parser.add_argument("--single_sided", action="store_true", help="The scans show fronts only")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help="Scan resolution in dpi when driving the scanner")
    parser.add_argument("--device", default=None,
                        help="SANE device name of the scanner (see: scanimage -L); default: the first one found")
    parser.add_argument("--no_driver_crop", action="store_true",
                        help="Do not ask the scanner driver to straighten and crop the cards")
    parser.add_argument("--scanimage_option", action="append", default=[],
                        help="Extra option passed on to scanimage, e.g. --scanimage_option=--depth=8 (repeatable)")
    args = parser.parse_args(argv)

    try:
        if args.from_folder is not None:
            rows = import_folder(args.from_folder, args.library, args.batch, args.single_sided)
        else:
            rows = scan_into_library(args.library, args.batch, args.resolution, args.device, args.single_sided,
                                     not args.no_driver_crop, tuple(args.scanimage_option))
    except CaptureError as error:
        raise SystemExit(str(error)) from error

    if not rows:
        raise SystemExit("No cards were scanned")
    to_review = [row for row in rows if row["status"] == STATUS_REVIEW]
    print(f"{len(rows)} card(s) added to batch {args.batch}: {rows[0]['card_id']} to {rows[-1]['card_id']}")
    if to_review:
        print(f"{len(to_review)} need a look (see the note column of {args.library / CATALOGUE_FILE}), "
              f"for example {to_review[0]['card_id']}: {to_review[0]['note']}")


if __name__ == "__main__":
    main()
