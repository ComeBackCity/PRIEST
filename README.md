# Influenza Project

Predicting mutations of influenza sequences


## Installation

```
# Run this in the project root
$ python3 -m venv env
$ source env/bin/activate
(env) $ pip install -r requirements.txt
```

## Usage

Scripts in the folder `./src/scripts/` are where you can run python code training.py directly.

To import scripts do `from src.<folder-name> import <file-name>`

You can also use the notebooks - after installation run `jupyter notebook` and make sure to run this at the top to enable auto reload of the scripts:

```
%load_ext autoreload
%autoreload 2
```

## Project Structure

    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── figure         <- Intermediate data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        │
        ├── code        <- Scripts for validation and visualization
        │
        ├── utils          <- Scripts for grouping scripts below into handy chunks
        │   └── utils.py
        │
		├── scripts       <- training and testing the dataset		
		│	   └── training.py
		│
        ├── data           <- Scripts to read in or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for processing
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models
        │   │                 
        │   └── train_model.py
        │   
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
		
--------

