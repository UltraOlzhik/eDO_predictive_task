import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(raw_data_path, processed_data_path, resample_freq='h'):
    """
    Process raw data to prepare it for feature engineering and model training.
    
    Args:
        raw_data_path (str): Path to the raw data CSV file.
        processed_data_path (str): Path where the processed data will be saved.
        resample_freq (str): Resampling frequency (default is 'H' for hourly).
    """
    try:
        # Load raw data
        df = pd.read_csv(raw_data_path, header=None)
        df.columns = ['TimeStamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df.set_index('TimeStamp', inplace=True)  # Set the timestamp as index for resampling
        
        # Resample the data
        df_resampled = df.resample(resample_freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
        }).dropna()

        logging.info(f'\nProcessed data:\n{df_resampled.head()}')

        # Save processed data
        df_resampled.to_csv(processed_data_path, index=True)
        logging.info(f'The processed data was saved to: {processed_data_path}')
    
    except FileNotFoundError as e:
        logging.error(f'File not found: {e}')
    except pd.errors.EmptyDataError as e:
        logging.error(f'No data: {e}')
    except Exception as e:
        logging.error(f'An error occurred: {e}')

# Example usage
# process_data('data/raw/historical_data.csv', 'data/processed/processed_data.csv')