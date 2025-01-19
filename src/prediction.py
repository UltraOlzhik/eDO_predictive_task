import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path):
    """
    Load the trained model from a file.
    
    Args:
        model_path (str): Path to the trained model file.
    
    Returns:
        model: The loaded model.
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        return None

def make_predictions(model, data_path, output_path):
    """
    Make predictions using the trained model.
    
    Args:
        model: The trained model.
        data_path (str): Path to the data with features CSV file.
        output_path (str): Path where the predictions will be saved.
    """
    try:
        # Load the data
        df = pd.read_csv(data_path, parse_dates=['TimeStamp'], index_col='TimeStamp')
        X = df.drop(columns=['Target'], errors='ignore')
        
        # Make predictions
        predictions = model.predict(X)
        df['Predictions'] = predictions
        
        # Save predictions
        df.to_csv(output_path)
        logging.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")

# Example usage
if __name__ == "__main__":
    model_path = 'models/trained_model.pkl'
    data_path = 'data/processed/processed_data_with_features.csv'
    output_path = 'results/predictions.csv'
    
    # Load the model
    model = load_model(model_path)
    
    if model is not None:
        make_predictions(model, data_path, output_path)