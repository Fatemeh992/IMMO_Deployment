import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import os
import joblib  # For saving and loading models
import pickle 

class XGBRegressionPipeline:
    """
    A pipeline for building and evaluating an XGBoost regression model,
    including model saving and loading.

    Parameters:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target variable in the dataset.
        numeric_feats (list): List of non-categorical feature names in the dataset.
        model_path (str): Path to save or load the model.
    """
    def __init__(self, data_path, target_column, numeric_feats, 
                 model_path="models/xgb_model.pkl"):
        self.data_path = data_path
        self.target_column = target_column
        self.numeric_feats = numeric_feats
        self.model_path = model_path
        self.model = None  #Placeholder for the model

    def load_data(self):
        """Loads the dataset from the specified path and separates it into features and target."""
        start_time = time.time()
        self.df = pd.read_csv(self.data_path)
        self.y = self.df[self.target_column]
        self.X = self.df.drop(self.target_column, axis=1)
        end_time = time.time()
        print(f"Data loading time: {end_time - start_time:.2f} seconds")
        return self.X, self.y

    def split_data(self, test_size=0.2):
        """Splits the dataset into training and testing sets."""
        start_time = time.time()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        end_time = time.time()
        print(f"Data splitting time: {end_time - start_time:.2f} seconds")

    def preprocess_data(self):
        """Preprocesses the data by applying log transformation and normalization."""
        start_time = time.time()
        numeric_feats = ['Living Area', 'Terrace area']
        #numeric_feats = [col for col in self.X_train.columns if col not in self.categorical_features]
        self.X_train[numeric_feats] = np.log1p(self.X_train[numeric_feats])
        self.X_test[numeric_feats] = np.log1p(self.X_test[numeric_feats])

        self.scaler = MinMaxScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=self.X_train.columns
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=self.X_test.columns
        )
        end_time = time.time()
        print(f"Data preprocessing time: {end_time - start_time:.2f} seconds")

    def save_model(self):
        """Saves the trained model to the specified path."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        pickle.dump(self.model, open(self.model_path, "wb"))
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the model from the specified path."""
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, "rb"))
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"No model found at {self.model_path}. Training a new model.")

    def perform_hyperparameter_tuning(self):
        """Performs GridSearchCV to find the best hyperparameters for the XGBoost model."""
        start_time = time.time()
        print("\nPerforming Hyperparameter Tuning...")
        params = {
            'objective': ['reg:squarederror'], 
            'max_depth': [6, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [300, 400],
            'subsample': [0.4, 0.8],
        }
        grid_mse = GridSearchCV(
            estimator=xgb.XGBRegressor(), 
            param_grid=params, 
            scoring='neg_mean_squared_error', 
            cv=4, 
            verbose=1
        )
        grid_mse.fit(self.X, self.y)
        end_time = time.time()
        print(f"Best Parameters: {grid_mse.best_params_}")
        print(f"Best RMSE: {np.sqrt(np.abs(grid_mse.best_score_)):.4f}")
        print(f"Hyperparameter tuning time: {end_time - start_time:.2f} seconds")
        self.model = grid_mse.best_estimator_

    def train_model(self):
        """Trains the XGBoost model on the training dataset."""
        start_time = time.time()
        if not self.model:
            raise ValueError("Model has not been initialized. Perform hyperparameter tuning first.")
        self.model.fit(self.X_train_scaled, self.y_train)
        end_time = time.time()
        print(f"Model training time: {end_time - start_time:.2f} seconds")

    def evaluate_model(self):
        """Evaluates the model using various metrics."""
        start_time = time.time()
        y_test_pred = self.model.predict(self.X_test_scaled)
        metrics = {
            "test": {
                "MAE": mean_absolute_error(self.y_test, y_test_pred),
                "RMSE": np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                "R2": r2_score(self.y_test, y_test_pred) * 100
            }
        }
        print(f"Model evaluation time: {time.time() - start_time:.2f} seconds")
        return metrics, y_test_pred

    def run(self):
        """Executes the entire pipeline: data loading, preprocessing, training, evaluation."""
        start_time = time.time()
        self.load_data()
        self.split_data()
        self.preprocess_data()
        self.load_model()
        if not self.model:
            self.perform_hyperparameter_tuning()
            self.train_model()
            self.save_model()
        metrics, y_test_pred = self.evaluate_model()
        print("\nTesting Metrics:")
        for metric, value in metrics["test"].items():
            print(f"{metric}: {value:.4f}")
        print(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds")

# Usage 
if __name__ == "__main__":
    numeric_feats = ['Living Area', 'Terace area']
    pipeline = XGBRegressionPipeline(data_path='Preprocessing/processed_data.csv',
        target_column='Price',numeric_feats=numeric_feats)
    pipeline.run()
