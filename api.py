from fastapi import FastAPI
from pydantic import BaseModel
import json
import pickle
import math
from Predict.prediction import predict 

app = FastAPI()

class PropertyInput(BaseModel):
    Locality: str
    TerraceArea: float
    LivingArea: float
    NumberOfFacades: int
    Fireplace: bool
    SwimmingPool: bool
    FullyEquippedKitchen: bool
    Furnished: bool
    PropertyType: str  
    PropertyState: str  


def property_subtype(subtype: str):
    dic = {'d_APARTMENT_BLOCK':0 ,'d_COUNTRY_COTTAGE':0 ,'d_DUPLEX':0,
             'd_EXCEPTIONAL_PROPERTY':0, 'd_FLAT_STUDIO':0, 'd_GROUND_FLOOR':0,
             'd_HOUSE':0, 'd_KOT':0 ,'d_MANSION':0,'d_MIXED_USE_BUILDING':0,
             'd_PENTHOUSE':0,'d_TOWN_HOUSE':0, 'd_VILLA':0,'d_other property':0}
    if subtype:
        if subtype == "other property":
            dic[f"d_{subtype}"] = 1
        else:
            dic[f"d_{'_'.join(subtype.split())}"] = 1
    return dic    

def property_state(state: str):
    dic = {'d_GOOD':0, 'd_JUST_RENOVATED':0,'d_TO_BE_DONE_UP':0,
         'd_TO_RENOVATE':0, 'd_TO_RESTORE':0}
    if state:
        dic[f"d_{'_'.join(state.split())}"] = 1
    return dic

@app.post("/predict/")
async def predict_price(input_data: PropertyInput):
    """
    Predict the property price based on the input data.
    """
    input_dict = input_data.model_dump()
    model_input = {
        'Locality': input_dict["Locality"],
        'Terrace area': input_dict["TerraceArea"], 
        'Living Area': input_dict["LivingArea"], 
        'Number of facades': input_dict["NumberOfFacades"],
        'Fireplace':input_dict["Fireplace"],
        'Swimming pool': input_dict["SwimmingPool"],
        'Fully equipped kitchen': input_dict["FullyEquippedKitchen"],
        'Furnished': input_dict["Furnished"],
    }

    # Convert boolean features to integers (0 or 1)
    model_input['Fireplace'] = int(model_input['Fireplace'])
    model_input['Swimming pool'] = int(model_input['Swimming pool'])
    model_input['Fully equipped kitchen'] = int(model_input['Fully equipped kitchen'])
    model_input['Furnished'] = int(model_input['Furnished'])

    model_input.update(property_subtype(input_dict["PropertyType"]))
    model_input.update(property_state(input_dict["PropertyState"]))
    # Model path and expected features
    model_path = 'models/xgb_model.pkl'
    expected_features = ['Locality', 'Terrace area', 'Living Area', 'Number of facades', 
                         'Fireplace', 'Swimming pool', 'Fully equipped kitchen', 'Furnished',
                         'd_APARTMENT_BLOCK', 'd_COUNTRY_COTTAGE', 'd_DUPLEX', 'd_EXCEPTIONAL_PROPERTY',
                         'd_FLAT_STUDIO', 'd_GROUND_FLOOR', 'd_HOUSE', 'd_KOT', 'd_MANSION', 'd_MIXED_USE_BUILDING',
                         'd_PENTHOUSE', 'd_TOWN_HOUSE', 'd_VILLA', 'd_other property', 'd_GOOD', 'd_JUST_RENOVATED',
                         'd_TO_BE_DONE_UP', 'd_TO_RENOVATE', 'd_TO_RESTORE']

    # Call your prediction function
    result = predict(input_data=model_input, model_path=model_path, expected_features=expected_features)
    print(result)

    return {"prediction": str(result)}
