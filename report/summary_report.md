# Summary Report: Predicting Target in H1 Time Frame

## Project Overview

The objective of this project is to predict the target category for the Close price in the next 10 H1 (1-hour) candles for November 2024 using historical M1 (1-minute) OHLCV data. The target categories are defined as follows:

- A: Close price < -5%
- B: Close price between -5% and -2%
- C: Close price between -2% and +2%
- D: Close price between +2% and +5%
- E: Close price > +5%

## Data

Raw Data: The raw data file eDO_data_M1.csv contains historical OHLCV data in the M1 time frame. The table is complete, but as shown in the "Target Variable Distribution" chart in notebooks/data_distribution_chart.ipynb, the data is imbalanced.
Processed Data: The processed data file processed.csv contains the data after resampling to the H1 time frame and adding relevant features.

## Data Processing

Script: src/data_processing.py
Description: This script handles the conversion of M1 OHLCV data to H1 OHLCV data using resampling. It also deals with missing values and prepares the data for feature engineering.
Output: The processed data is saved as data/processed/processed_data.csv.

## Feature Engineering

Script: src/feature_engineering.py
Description: This script adds new features such as moving averages (e.g., MA_10) and other technical indicators to assist in predicting the target category.
Output: Feature-engineered data is saved back to data/processed/processed_data_with_features.csv.

## Model Training

Script: src/model_training.py
Description: This script trains a RandomForestClassifier on the processed data. The target variable is categorized based on the percentage change in the Close price over the next 10 H1 candles. The data is split into training and testing sets, and the model is trained.
Output: The trained model is saved as models/trained_model.pkl.

## Predictions

Script: src/predictions.py
Description: This script loads the trained model and makes predictions on the test data. The predictions are saved to results/predictions.csv.
Output: Predictions are saved in results/predictions.csv.

## Evaluation

Script: src/evaluation.py
Description: This script evaluates the model's performance using a confusion matrix and classification report. The confusion matrix is saved as an image file for visual representation.
Output: The confusion matrix is saved as results/confusion_matrix.png.

# Results

## Confusion Matrix

The confusion matrix indicates the performance of the model across different target categories. It shows the number of true positives, false positives, false negatives, and true negatives for each category.

## Classification Report

The classification report provides precision, recall, and F1-score for each category, along with overall accuracy, macro average, and weighted average.


| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| A        | 0.97      | 1.00   | 0.98     | 317     |
| B        | 0.93      | 0.99   | 0.96     | 1799    |
| C        | 1.00      | 0.99   | 0.99     | 21139   |
| D        | 0.94      | 0.99   | 0.96     | 2075    |
| E        | 0.98      | 1.00   | 0.99     | 356     |
| Accuracy     |           |        | 0.99     | 25686   |
| Macro avg    | 0.96      | 0.99   | 0.98     | 25686   |
| Weighted avg | 0.99      | 0.99   | 0.99     | 25686   |


# Conclusion

The model demonstrates high accuracy and strong performance across all target categories, with an overall accuracy of 99%. The precision, recall, and F1-scores are all very high, indicating that the model is reliable and effective in predicting the target categories. This performance suggests that the model can be used with confidence in predicting future price movements.

## Future considerations

- Check for Overfitting: Ensure the model is not overfitting by evaluating it on a separate validation set or using cross-validation.
- Feature Importance: Investigate the importance of each feature to understand their impact on the predictions.
- Analyze Misclassifications: Review the confusion matrix to identify patterns in misclassified instances and explore ways to improve the model further.

## Additional Resources

Data Distribution Chart: A notebook showing the distribution of the data is available in notebooks/data_distribution_chart.ipynb.
Requirements: The required libraries and dependencies are listed in requirements.txt.