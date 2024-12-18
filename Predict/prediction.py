import os
import joblib
import numpy as np
import pandas as pd
import pickle

def load_model(model_path):
    """
    Load the trained machine learning model from the specified path.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        sklearn.base.BaseEstimator: The loaded model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' does not exist.")
    model = pickle.load(open(model_path, "rb"))
    print(f"Model loaded successfully from {model_path}")
    return model

def preprocess_input(input_data, expected_features):
    """
    Ensures the input data is in the correct format for prediction.
    Args:
        input_data (dict or pd.DataFrame): The new house data to predict on.
        expected_features (list): List of expected feature names in the correct order.
    Returns:
        np.ndarray: A 2D array with the preprocessed features.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("Input data must be a dictionary or a pandas DataFrame.")
    
    # Ensure the input has the required features
    print(type(expected_features))
    print(expected_features)
    print(input_df.columns)
    missing_features = [feature for feature in expected_features if feature not in input_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Ensure the features are in the correct order
    input_df = input_df[expected_features]
    
    return input_df.values  # Return as a NumPy array

def predict(input_data, model_path, expected_features):
    """
    Predict the price of a new house based on preprocessed data.
    
    Args:
        input_data (dict or pd.DataFrame): The input data representing a new house.
        model_path (str): Path to the trained machine learning model file.
        expected_features (list): List of feature names expected by the model.
    
    Returns:
        float: The predicted price of the house.
    """
    # Load the trained model
    model = load_model(model_path)
    # Preprocess the input data
    processed_input = preprocess_input(input_data, expected_features)
    # Perform prediction
    predicted_price = model.predict(processed_input)
    
    return predicted_price[0]  # Return the first prediction (assumes one house at a time)

if __name__ == "__main__":
    # Example usage:
    MODEL_PATH = "models/xgb_model.pkl"
    EXPECTED_FEATURES = ['Locality','Terrace area', 'Living Area', 'Number of facades',
                                    'Fireplace', 'Swimming pool',
                                    'd_APARTMENT_BLOCK','d_COUNTRY_COTTAGE','d_DUPLEX',
                                    'd_EXCEPTIONAL_PROPERTY', 'd_FLAT_STUDIO', 'd_GROUND_FLOOR',
			 'd_HOUSE', 'd_KOT' ,'d_MANSION', 'd_MIXED_USE_BUILDING','d_PENTHOUSE',
			 'd_TOWN_HOUSE', 'd_VILLA', 'd_other property', 'd_GOOD', 'd_JUST_RENOVATED',
			 'd_TO_BE_DONE_UP', 'd_TO_RENOVATE', 'd_TO_RESTORE', 'Fully equipped kitchen',
             'Furnished']

  
    # Example new house data
    new_house_data = {
        "Living Area": 120,
        "Number of facades": 4,
        "Terrace area": 20,
        "Locality": 50,
        "Fireplace": 1,
        "Swimming pool": 0,
        "d_APARTMENT_BLOCK":1,
        "d_COUNTRY_COTTAGE":0,
        "d_DUPLEX":0,
        "d_EXCEPTIONAL_PROPERTY":0,
        "d_FLAT_STUDIO":0, 
        "d_GROUND_FLOOR":0,
		"d_HOUSE":0, 
        "d_KOT":0 ,
        "d_MANSION":0,
        "d_MIXED_USE_BUILDING":0,
        "d_PENTHOUSE":0,
	    "d_TOWN_HOUSE":0,
        "d_VILLA":0, 
        "d_other property":0,
        "d_GOOD":1,
        "d_JUST_RENOVATED":0,
	    "d_TO_BE_DONE_UP":0, 
        "d_TO_RENOVATE" : 0,
        "d_TO_RESTORE" : 0,
        "Fully equipped kitchen" : 1,
        "Furnished" : 0

    }
    
    try:
        # Predict the house price
        predicted_price = predict(new_house_data, MODEL_PATH, EXPECTED_FEATURES)
        print(f"The predicted price of the house is: €{predicted_price:.2f}")
    except Exception as e:
        print(f"Error: {e}")

