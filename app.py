import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 
import json
import math
import requests 

def welcome(): 
    return 'Welcome Dear Client!'

def main(): 
    st.title("Property Price Prediction") 

    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit XGBoost ML App </h1> 
    </div> 
    """
    
    locality = json.load(open("locality.json"))
    st.markdown(html_temp, unsafe_allow_html=True) 

    # User inputs
    Property_Locality = st.selectbox("Locality", locality) 
    Property_LivingArea = st.number_input("Living Area", placeholder=125)
    Property_TerraceArea = st.number_input("Terrace Area", placeholder=80) 
    Property_facades = st.number_input("Number of facades", placeholder=2) 
    Property_Fireplace = st.checkbox("Fireplace", False)
    Property_SwimmingPool = st.checkbox("Swimming Pool", False)
    Property_kitchen = st.checkbox("Fully equipped kitchen", False)
    Property_furniture = st.checkbox("Furnished", False)
    Property_type = st.selectbox('Select a feature value:',
                    ['APARTMENT BLOCK','COUNTRY COTTAGE','DUPLEX',
                    'EXCEPTIONAL PROPERTY', 'FLAT STUDIO', 'GROUND FLOOR',
                    'HOUSE', 'KOT' ,'MANSION', 'MIXED USE BUILDING','PENTHOUSE',
                    'TOWN HOUSE', 'VILLA', 'other property'])
    Property_state = st.selectbox('Select a feature value:', 
                    ['GOOD', 'JUST RENOVATED','TO BE DONE UP',
                         'TO RENOVATE', 'TO RESTORE'])

    # Prepare data for prediction
    if st.button("Predict"):
        input_data = {
            'Locality': Property_Locality,
            'TerraceArea': Property_TerraceArea,
            'LivingArea': Property_LivingArea,
            'NumberOfFacades': int(Property_facades),
            'Fireplace': Property_Fireplace,
            'SwimmingPool': Property_SwimmingPool,
            'FullyEquippedKitchen': Property_kitchen,
            'Furnished': Property_furniture,
        }

        # Add PropertyType and PropertyState to input data
        input_data.update({"PropertyType": Property_type})
        input_data.update({"PropertyState": Property_state})

        print(input_data)
        # Make an API call to the FastAPI backend
        api_url = "http://127.0.0.1:8000/predict/"
        response = requests.post(api_url, json=input_data)

        if response.status_code == 200:
            prediction_result = response.json()["prediction"]
            st.success(f"The predicted property price is: {prediction_result}")
        else:
            st.error("Error in prediction. Please try again.")

if __name__ == '__main__':
    main()
