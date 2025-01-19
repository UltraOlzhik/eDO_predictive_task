# H1 Time Frame Prediction Project

Predicting Price Movements from Historical OHLCV Data

## Overview

This project aims to predict price movements in the H1 time frame based on historical OHLCV data from the M1 time frame. The process includes data preprocessing, feature engineering, model training, evaluation, and generating predictions.

## Directory Structure

```sh
eDO_predictive_task/
├── data/
│   ├── raw.csv
│   ├── processed.csv
├── models/
│   └── trained_model.py
├── notebooks/
│   └── data_distribution_chart.ipynb
├── report/
│   ├── summary_report.md
│   └── requirements.txt
├── results/
│   ├── confusion_matrix.png
│   └── predictions.csv
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predictions.py
│   └── evaluation.py
├── main.py
└── README.md
```

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/UltraOlzhik/eDO_predictive_task.git
    cd eDO_predictive_task
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```sh
    pip install -r report/requirements.txt
    ```
4. The directory data/raw is empty. Please add the original eDO_data_M1.csv file there and then proceed.

## Data Processing
The data processing pipeline is handled by the `data_processing.py` script. It converts M1 OHLCV data to H1 OHLCV data using resampling and prepares it for feature engineering.

## Feature Engineering
The `feature_engineering.py` script adds new features such as moving averages and other technical indicators to aid in prediction.

## Model Training
The `model_training.py` script trains a `RandomForestClassifier` on the processed data. The target variable is categorized based on the percentage change in the Close price over the next 10 H1 candles.

## Predictions
The `predictions.py` script loads the trained model and makes predictions on the test data. The predictions are saved to `results/predictions.csv`.

## Evaluation
The `evaluation.py` script evaluates the model's performance using a confusion matrix and classification report. The confusion matrix is saved as an image file.

## Running the Pipeline
To run the complete pipeline, execute the `main.py` script:
```sh
python main.py
```

# Summary Report

For a detailed summary of the project, refer to the summary_report.md file in the report directory.
