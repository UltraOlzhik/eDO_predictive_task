import src.data_processing as dp
import src.feature_engineering as fe
import src.model_training as mt
import src.prediction as pr
import src.evaluation as ev
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting the predictive task project pipeline...")
    
    try:
        # Step 1: Data Preprocessing
        logging.info("Step 1: Data Preprocessing")
        dp.process_data(
            raw_data_path='data/raw/eDO_data_M1.csv', 
            processed_data_path='data/processed/processed_data.csv'
        )
        
        # Step 2: Feature Engineering
        logging.info("Step 2: Feature Engineering")
        df = fe.load_features('data/processed/processed_data.csv')
        df = fe.add_rolling_features(df, [10, 20, 50])
        df = fe.add_price_change_features(df)
        df = fe.add_target_column(df, future_candles=10)
        fe.save_features(df, 'data/processed/processed_data_with_features.csv')
        
        # Step 3: Model Training
        logging.info("Step 3: Model Training")
        df = fe.load_features('data/processed/processed_data_with_features.csv')
        X, y = mt.preprocess_data(df)
        X_resampled, y_resampled = mt.handle_imbalanced_data(X, y)
        X_train, X_test, y_train, y_test = mt.train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        model = mt.train_model(X_train, y_train)
        mt.save_model(model, 'models/trained_model.pkl')
        
        # Step 4: Prediction
        logging.info("Step 4: Prediction")
        pr.make_predictions(
            model, 
            data_path='data/processed/processed_data_with_features.csv', 
            output_path='results/predictions.csv'
        )
        
        # Step 5: Evaluation
        logging.info("Step 5: Evaluation")
        df_predictions = ev.load_predictions('results/predictions.csv')
        evaluation_results = ev.evaluate_predictions(df_predictions)
        ev.plot_confusion_matrix(evaluation_results['confusion_matrix'], 'results/confusion_matrix.png')
        
        logging.info("Pipeline completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred in the pipeline: {e}")

if __name__ == "__main__":
    main()