<h1 align="center">Automated Grading of Baseball Trading Cards using Convolutional Neural Networks</h1>  

### Authors:  
- Brian Miller (brian.s.miller@du.edu)
- Sean Wendlandt (sean.wendlandt@du.edu)

### Project Overview:
The condition of a trading card significantly influences its worth. However, the current methods to assess a card's condition are insufficient. Non-professionals often lack the accuracy required for card grading, and obtaining professional evaluations is too expensive and time-consuming.

This project aims to automate the card grading process using machine learning. Our neural network utilizes the pretained model ResNet-50 
combined with custom classification and output layers. 

**Data Preprocessing Steps:**
- Web scraping to obtain data
- Manual removal of innacurate data
- Dataset balancing
- Automated removal of PSA label
- Standardize photo dimensions for CNN

### Description of CNN:
![alt text](readme_photos/photo_diagram.png)
- ResNet-50 (global average pooled) feeds into three fully connected ReLU dense layers [1024, 512, 128]  
- The dense layers feed into a single softmax prediction layer  

Two rounds of training were done:
- First Round: 50 epochs, only the dense layers and the last 20 layers of ResNet-50 are trained
- Second Round: 50 epochs, the last 50 layers of ResNet-50 are unfrozen and fine-tuned at a 10x lower learning rate

### Setup:
The code is one installable Python package, `card_grading` (Python 3.10+):
```
pip install -e ".[all]"
```
Smaller installs are available: `.[train]` (TensorFlow), `.[scrape]`, `.[notebooks]` and `.[dev]` (tests and linting).

### Usage:
Installing the package adds one command per step of the pipeline:

| Step | Command |
| --- | --- |
| 1. Scrape images and metadata for one PSA grade | `card-grading-scrape --grade 1 --files_dir ./collection_files --photo_dir ./collection_files/images_psa1` |
| 2. Crop the PSA label off and standardize the size | `card-grading-crop --parent_dir ./final_10k --output_dir ./final10k_cropped` |
| 3. Add flipped copies of the training images | `card-grading-flip ./data/train` |
| 4. Train and evaluate the model | `card-grading-train --n_epochs 50 --droprate 0.025 --base_path ./data` |

Run any command with `--help` for all of its options. Each training run writes its training log, parameters
and saved models to `logs/<timestamp>-<pid>/`.

### Training on a public dataset:
The original images are not public, and Collectors.com no longer offers the search they were scraped from.
[PSA-Baseball-Grades](https://github.com/samsilverman/PSA-Baseball-Grades) is a public set of 11,500 slab photos
(1,150 per PSA grade, 150x200 pixels) that the pipeline can be run on:
```
git clone --depth 1 https://github.com/samsilverman/PSA-Baseball-Grades
card-grading-crop --parent_dir PSA-Baseball-Grades --output_dir data --target_width 150
card-grading-train --base_path data --image_height 160 --image_width 160 --batch_size 32 --n_epochs 10 --early_stopping_patience 3
```
Things to know about that dataset: its repository states no license, its author warns that some grades are wrong
and some images are irrelevant, and it contains exact duplicate images (the train/test split sets those aside in
`data/duplicates/`).

### Development:
```
ruff check .
pytest
```

### Data:
All 10k cards were downloaded from [Collectors.com](https://www.collectors.com/trading-cards/sport-baseball-cards/20003?lowgrade=1&highgrade=10&gradingservice=2&page=1)

### Description of Files:
```
pyproject.toml            package metadata, dependencies and tool settings
src/card_grading/
    data.py               PSA grades, file naming and the train/test split
    scraping.py           web scraper for Collectors.com
    cropping.py           PSA label removal and image resizing
    augmentation.py       flipped copies of the training images
    training.py           ResNet-50 model, training and evaluation
scripts/                  shell scripts that run the commands in bulk (dropout sweep, grid search, all grades)
notebooks/                metadata building, dataset balancing and EDA
tests/                    unit tests
testing_data_folder/      a small sample of the dataset, in the layout the training expects
```
