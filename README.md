This repository contains code and resources for predicting product demand using machine learning techniques. The project aims to help businesses optimize inventory and 
supply chain management by forecasting future product demand based on historical data.

## Table of Contents
1.Requirements
2.Installation
3.Data
4.Usage
5.Model Training
6.Inference
7.Results

## Requirements
Before running the code, ensure you have the following dependencies installed:

Python 3.x
Jupyter Notebook (for running the provided notebooks)
pandas
numpy
scikit-learn
matplotlib (for data visualization)
seaborn (for data visualization)
Any additional libraries or dependencies as specified in the project's environment.yml or requirements.txt files.

## Installation
Clone the repository:

git clone https:https://github.com/ShaliniKanniyappan/ADS_phase1.git

## Data
The project relies on historical sales and product information data for training and testing the model. You should provide your dataset or use a sample dataset available 
in the data/ directory. Ensure that your data is in a CSV or Excel format and contains the following columns:

Here we take dataset from kagggle
url link for dataset:https://www.kaggle.com/datasets/chakradharmattapalli/product-demand-prediction-with-machine-learning 

Source
The dataset used in this project is publicly available on [Kaggle](https://www.kaggle.com/dataset/source). You can download the dataset from the provided link, and it contains historical sales and product information.

Description
The dataset consists of the following columns:

- `date`: Date of the sales transaction
- `product_id`: Unique identifier for the product
- `quantity_sold`: The quantity of the product sold on that date

Please note that the dataset source may change over time, so make sure to provide the most up-to-date source in your README.

## Usage
The project consists of two main phases: model training and inference. Follow the instructions below for each phase:

## Model Training
Open the Jupyter Notebook for model training:

jupyter notebook model_training.ipynb
Follow the instructions in the notebook to train your demand prediction model using the provided data.

## Inference
Open the Jupyter Notebook for making predictions:

jupyter notebook inference.ipynb
Provide the necessary input data (e.g., future dates and product IDs) and run the notebook to make predictions.

## Results
The trained model's predictions will be available in the inference notebook, and you can visualize the results using the provided data visualization tools.
