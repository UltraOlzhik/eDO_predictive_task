import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
import logging

# Set the number of CPU cores to use
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df):
    """
    Preprocess the data by separating features and target, and handling missing values.
    
    Args:
        df (pd.DataFrame): DataFrame containing the feature-engineered data.
    
    Returns:
        pd.DataFrame, pd.Series: Features (X) and target (y).
    """
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns

    # Impute missing values in numeric columns
    imputer = SimpleImputer(strategy='mean')
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # Drop non-numeric columns
    X.drop(columns=non_numeric_cols, inplace=True)

    return X, y

def handle_imbalanced_data(X, y):
    """
    Handle imbalanced data using SMOTE.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
    
    Returns:
        pd.DataFrame, pd.Series: Resampled features (X_resampled) and target (y_resampled).
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

def train_model(X_train, y_train):
    """
    Train a RandomForest classifier.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def save_model(model, model_path):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained model.
        model_path (str): Path to save the model.
    """
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

# Example usage
if __name__ == "__main__":
    logging.info("Step 3: Model Training")
    df = pd.read_csv('data/processed/processed_data_with_features.csv', parse_dates=['TimeStamp'], index_col='TimeStamp')
    
    try:
        X, y = preprocess_data(df)
        X_resampled, y_resampled = handle_imbalanced_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        save_model(model, 'models/trained_model.pkl')
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")