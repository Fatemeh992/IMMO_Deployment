# Property Price Prediction App

This project provides a full-stack solution for predicting property prices in Belgium using a machine learning model as part of a project at Becode in 2024. It includes two components:

1. **FastAPI Backend**: Handles prediction requests and runs the trained model.
2. **Streamlit Frontend**: Provides a user interface for inputting property details and displaying predictions.

---

## Features

- User-friendly interface to input property details.
- Real-time predictions using a trained XGBoost model.
- Backend powered by FastAPI for efficient API responses.
- Frontend built with Streamlit for an interactive experience.

---

## Technologies Used

### Backend
- **FastAPI**: API framework for handling prediction requests.
- **Python**: Core language for data processing and prediction.
- **XGBoost**: Machine learning model for property price prediction.

### Frontend
- **Streamlit**: Framework for creating the user interface.
- **Pandas & NumPy**: Data manipulation libraries.

---

## Setup and Installation

### Prerequisites

1. Python 3.8 or higher installed on your machine.
2. Required Python libraries:
   - fastapi
   - uvicorn
   - streamlit
   - pandas
   - numpy
   - xgboost
   - requests

Install all dependencies by running:
```bash
pip install -r requirements.txt
```

### Folder Structure
```
.
├── FastAPI_Backend
│   ├── api.py                # FastAPI application
│   ├── models
│   │   └── xgb_model.pkl     # Trained XGBoost model
│   └── Predict
│       └── prediction.py     # Prediction logic
├── Streamlit_Frontend
│   ├── streamlit_app.py      # Streamlit app
│   └── locality.json         # Locality dropdown options
└── README.md                 # Project documentation
```

---


## How to Use

1. Enter property details in the input fields (e.g., locality, living area, terrace area, etc.).
2. Click the **Predict** button.
3. View the predicted property price displayed on the screen.

---

## API Documentation

The FastAPI backend provides an endpoint for property price predictions:

- **POST** `/predict/`
  - **Input**: JSON object containing property details.
  - **Output**: JSON object with the predicted property price.

Example Request:
```json
{
  "Locality": "Brussels",
  "TerraceArea": 50,
  "LivingArea": 100,
  "NumberOfFacades": 2,
  "Fireplace": true,
  "SwimmingPool": false,
  "FullyEquippedKitchen": true,
  "Furnished": false,
  "PropertyType": {
    "d_APARTMENT_BLOCK": 1,
    "d_HOUSE": 0
  },
  "PropertyState": {
    "d_GOOD": 1,
    "d_TO_RENOVATE": 0
  }
}
```

---

## Model Details

The model used for prediction is an XGBoost regressor trained on property datasets. The target variable is the price.

---

## Future Improvements

- Add support for more property features.
- Enhance the UI for a better user experience.
- Optimize the machine learning model for higher accuracy.

---

## License

This project is open-source.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Contact

For questions or suggestions, contact [Fatemeh992].

