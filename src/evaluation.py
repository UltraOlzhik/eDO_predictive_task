import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_predictions(predictions_path):
    """
    Load the predictions from a CSV file.
    
    Args:
        predictions_path (str): Path to the predictions CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with predictions.
    """
    try:
        df = pd.read_csv(predictions_path, parse_dates=['TimeStamp'], index_col='TimeStamp')
        logging.info(f"Predictions loaded from {predictions_path}")
        return df
    except Exception as e:
        logging.error(f"An error occurred while loading the predictions: {e}")
        return None

def evaluate_predictions(df, target_col='Target', pred_col='Predictions', proba_col_prefix='Proba_'):
    """
    Evaluate the predictions using various metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing the true labels and predictions.
        target_col (str): Column name for the true labels.
        pred_col (str): Column name for the predicted labels.
        proba_col_prefix (str): Prefix for the probability columns for each class.
    
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    try:
        y_true = df[target_col]
        y_pred = df[pred_col]
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=['A', 'B', 'C', 'D', 'E'])
        logging.info(f"Classification report:\n{class_report}")
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        logging.info(f"Confusion matrix:\n{conf_matrix}")
        
        # ROC-AUC score for multi-class classification
        proba_cols = [col for col in df.columns if col.startswith(proba_col_prefix)]
        if len(proba_cols) > 0:
            y_pred_proba = df[proba_cols].values
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro')
            logging.info(f"ROC-AUC Score: {roc_auc}")
        else:
            roc_auc = None
        
        return {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc
        }
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        return None

def plot_confusion_matrix(conf_matrix, output_path):
    """
    Plot and save the confusion matrix.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        output_path (str): Path to save the confusion matrix plot.
    """
    try:
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['A', 'B', 'C', 'D', 'E'], yticklabels=['A', 'B', 'C', 'D', 'E'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Confusion matrix plot saved to {output_path}")
    except Exception as e:
        logging.error(f"An error occurred while plotting the confusion matrix: {e}")

# Example usage
if __name__ == "__main__":
    predictions_path = 'results/predictions.csv'
    confusion_matrix_path = 'results/confusion_matrix.png'
    
    # Load predictions
    df = load_predictions(predictions_path)
    
    if df is not None:
        # Evaluate predictions
        evaluation_results = evaluate_predictions(df)
        
        if evaluation_results is not None:
            # Plot confusion matrix
            plot_confusion_matrix(evaluation_results['confusion_matrix'], confusion_matrix_path)