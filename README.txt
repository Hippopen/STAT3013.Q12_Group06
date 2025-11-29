STAT3013.Q12_Group06 – Model Collection
======================================

1. Overview
-----------

This repository contains code and Jupyter notebooks for a STAT3013 group project.
The goal is to build and compare different regression / time-series models on
the same dataset.

The main folders correspond to different model families:
- linear_regression_model  : baseline linear regression models
- RandomForest_code        : Random Forest models
- XGBoost_code             : XGBoost models
- LightGBM                 : LightGBM models
- CatBoost                 : CatBoost models
- SARIMAX_code             : SARIMAX time-series models
- Prophet_code             : Prophet time-series models
- LSTM                     : LSTM-based neural network models

Most work is implemented in .ipynb notebooks, with some helper .py scripts.


2. Environment requirements
---------------------------

Recommended:
- Python 3.9 – 3.11
- Jupyter Notebook or JupyterLab

Core Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Time-series and statistical modelling:
- statsmodels
- prophet  (or fbprophet, depending on the installation)

Gradient boosting and tree-based models:
- xgboost
- lightgbm
- catboost

Deep learning for LSTM:
- tensorflow (with keras) OR pytorch
  (check the imports in the LSTM notebook and install the one that is used)

Example of environment setup with pip:

1) Create and activate a virtual environment

   python -m venv stat3013_env

   # Windows:
   stat3013_env\Scripts\activate

   # Linux / macOS:
   source stat3013_env/bin/activate

2) Upgrade pip and install packages

   pip install --upgrade pip
   pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
   pip install xgboost lightgbm catboost prophet jupyter
   pip install tensorflow
   # or, instead of tensorflow:
   # pip install torch


3. How to run the models
------------------------

There is no single "main" script for all models. Each model family is run
separately from its own folder.

Typical workflow:

1) Start Jupyter:

   jupyter notebook
   # or:
   jupyter lab

2) In the Jupyter interface, open the folder of the model you want to run,
   for example:
   - linear_regression_model
   - RandomForest_code
   - XGBoost_code
   - LightGBM
   - CatBoost
   - SARIMAX_code
   - Prophet_code
   - LSTM

3) Open the corresponding .ipynb notebook for that model.

4) Run all cells from top to bottom to:
   - load and preprocess the dataset,
   - train the model,
   - evaluate metrics and plot the results.

If you get a "ModuleNotFoundError", install the missing package in your
environment and re-run the notebook.


4. Data
-------

The notebooks assume that the dataset is available in the paths defined in the
first cells (data loading part). If you change the data location, please update
those paths before running the notebooks.


5. License
----------

This repository does NOT currently include an explicit open-source license.

That means:
- By default, all rights are reserved by the authors.
- The code is intended for course and project use.

If you want to reuse or redistribute the code, please:
- Contact the project members for permission, or
- Add an appropriate LICENSE file (for example: MIT, BSD-3-Clause, or GPL)
  and update this section.
