import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_rolling_features(df, window_sizes):
    """
    Add rolling statistical features (mean, std, etc.) to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        window_sizes (list): List of window sizes for rolling calculations.
    
    Returns:
        pd.DataFrame: DataFrame with added rolling features.
    """
    for window in window_sizes:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
        df[f'rolling_min_{window}'] = df['Close'].rolling(window=window).min()
        df[f'rolling_max_{window}'] = df['Close'].rolling(window=window).max()
    return df

def add_price_change_features(df):
    """
    Add price change and return features.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
    
    Returns:
        pd.DataFrame: DataFrame with added price change features.
    """
    df['price_change'] = df['Close'] - df['Open']
    df['price_change_pct'] = (df['price_change'] / df['Open']) * 100
    df['high_low_range'] = df['High'] - df['Low']
    df['high_low_range_pct'] = (df['high_low_range'] / df['Low']) * 100
    return df

def add_target_column(df, future_candles, thresholds=(-5, -2, 2, 5)):
    """
    Add the target column based on future price movement.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        future_candles (int): Number of steps (candles) to predict into the future.
        thresholds (tuple): Thresholds for categorizing percentage change.
    
    Returns:
        pd.DataFrame: DataFrame with added target column.
    """
    try:
        # Calculate the future price change percentage
        df['future_close'] = df['Close'].shift(-future_candles)
        df['future_change_pct'] = ((df['future_close'] - df['Close']) / df['Close']) * 100

        # Categorize the target based on thresholds
        def categorize_change(change):
            if change < thresholds[0]:
                return 'A'  # < -5%
            elif thresholds[0] <= change < thresholds[1]:
                return 'B'  # -5% to -2%
            elif thresholds[1] <= change < thresholds[2]:
                return 'C'  # -2% to +2%
            elif thresholds[2] <= change < thresholds[3]:
                return 'D'  # +2% to +5%
            else:
                return 'E'  # > +5%

        df['Target'] = df['future_change_pct'].apply(categorize_change)

        df = df.drop(['future_close', 'future_change_pct'], axis=1)

    except KeyError as e:
        logging.error(f'Missing column in DataFrame: {e}')
    except Exception as e:
        logging.error(f'An error occurred: {e}')

    return df

def save_features(df, file_path):
    """
    Save the feature-engineered DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features.
        file_path (str): Path to save the CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Feature-engineered data saved to {file_path}")
    except Exception as e:
        logging.error(f'An error occurred while saving the file: {e}')

def load_features(file_path):
    """
    Load the feature-engineered data from a CSV file.
    
    Args:
        file_path (str): Path to the feature file.
    
    Returns:
        pd.DataFrame: DataFrame with features.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Feature-engineered data loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f'An error occurred while loading the file: {e}')
        return None

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed/processed_data.csv', parse_dates=['TimeStamp'], index_col='TimeStamp')
    df = add_rolling_features(df, [10, 20, 50])
    df = add_price_change_features(df)
    df = add_target_column(df, future_candles=10)
    save_features(df, 'data/processed/processed_data_with_features.csv')