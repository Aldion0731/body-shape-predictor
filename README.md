### Overview

The goal of this project was to use machine learning to predict bust circumference, hip circumference and waist circumference from a dataset of 5137 labeled training examples. THe source code for the project is organized under a `src` folder in the root of the project folder.


### Organization of the `scr` folder

The `src` folder is divided into three sub-folders: `notebooks`, `steps` and `utils`.  

The `notebooks`folder contain 6 sequentially named `Jupyter notebooks`. The first of these overview the EDA process that was employed. The subsequent notebooks demonstrate the building and training of various machine learning models. Each notebook ends with the generation of predictions for the data in in the `test.csv` file. The generated predictions are then embedded in a copy of the serving data (from the original `test.csv` file), appropriately named and saved as a new csv file in the `predictions` folder of the project directory. 


The `steps` folder contain three modules which act as high level API's. These high level modules abstract away the implementation details which may be found in the various utility modules found in the `utils` folder.

`load_data.py` is responsible for loading and preprocessing data as well as feature engineering. This module provides functions which return data that is ready to be trained. By default, it splits the original training data (from `train.csv`) into training, validation and test splits, in the ratio 70:15:15. This ratio can be configured using the `config.toml` file located in the root of the project folder. Please see the `config.toml` file for additional configuration settings which may be modified by the user.

`train.py` is responsible for training models. It is a flexible API that just needs to be passed models, data, and desired model configurations. It is designed to accept the end results of the data generated from `load_data.py`

`predict.py` is responsible for generating predictions from the serving data (from the original `test.csv` file). It also embeds those predictions in the serving data dumps them to a new csv file in the prior mention `predictions` directory. 

The code in these modules is meant to be highly reusable and to support fast iterations.


### Steps to recreating the project

1.  Install `pipenv`

2. Install dependencies

```bash
pipenv sync
```
3. Run the code in the various notebooks located in the `src/notebooks` directory.


### Further Improvements

This project may be improved by further hyperparameter tuning and employing architecture search techniques for a better model architecture. For this phase of the project, developer time was mainly spent on optimizing the code for fast iterations.

Another future improvement will be the implementation of more flexibility in the design of the `FeatureEngineer` APi. This will allow for model specific feature engineering as opposed to being constrained to use identical feature engineering for each iteration.