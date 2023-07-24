### Overview

The goal of this project was to use machine learning to predict bust circumference, hip circumference and waist circumference from a dataset of 5137 labeled training examples. THe source code for the project is organized under a `src` folder in the root of the project folder.


### Organization of the `scr` folder

The src folder is divided into three sub-folders: `notebooks`, `steps` and `utils`.  

The `notebooks`folder contain 6 sequentially named `Jupter notebooks`. The first of these overview the EDA process that was employed. The subsequent notebooks demonstrate the building and training of various machine learning models. Each notebook ends with the generation of predictions for the data in in the `test.csv` file. The generated predictions are then embedded in a copy of the `test.csv` file, and saved as a new csv file. For the remainder of this document, the data in `test.csv` will often be referred to as serving data.

The `steps` folder contain three modules which act as high level API's. These high level modules abstract away the implementation details which may be found in the various utility modules found in the `utils` folder.

`load_data.py` is reponsible for loading and preprocessing data as well as feature engineering. This module provides functions which return data that is ready to be trained.

`train.py` is responsible for training models. It is a flexible API that just needs to be passed models, data and desired model configurations. It is designed to accept the end results of the data generated from `load_data.py`

`predict.py` is responsible for generating predictions from the serving data. It also embeds those predictions in the serving data dumps them to a new file similar to `train.csv` 

The code in these modules is meant to be highly reusable and to support fast iterations.
