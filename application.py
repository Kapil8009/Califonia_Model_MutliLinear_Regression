from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

application = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('model/housing_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    print("Model and scaler loaded successfully!")
except FileNotFoundError:
    print("Please run model.py first to train and save the model!")
    model = None
    scaler = None

# Feature names for California Housing dataset
FEATURE_NAMES = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]

FEATURE_DESCRIPTIONS = {
    'MedInc': 'Median income in block group',
    'HouseAge': 'Median house age in block group',
    'AveRooms': 'Average number of rooms per household',
    'AveBedrms': 'Average number of bedrooms per household',
    'Population': 'Block group population',
    'AveOccup': 'Average household occupancy',
    'Latitude': 'Block group latitude',
    'Longitude': 'Block group longitude'
}

@application.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES, 
                         descriptions=FEATURE_DESCRIPTIONS)

@application.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('result.html', 
                             error="Model not loaded. Please train the model first.")
    
    try:
        # Get values from the form
        features = []
        for feature in FEATURE_NAMES:
            value = float(request.form[feature])
            features.append(value)
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale the features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # California housing prices are in $100,000s
        predicted_price = prediction * 100000
        
        # Create input summary
        input_summary = dict(zip(FEATURE_NAMES, features))
        
        return render_template('result.html',
                             prediction=f"${predicted_price:,.2f}",
                             prediction_raw=prediction,
                             input_summary=input_summary,
                             features=FEATURE_NAMES,
                             descriptions=FEATURE_DESCRIPTIONS)
    
    except Exception as e:
        return render_template('result.html', 
                             error=f"Error making prediction: {str(e)}")

@application.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions"""
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        features = [float(data[feature]) for feature in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'predicted_price': float(prediction * 100000),
            'input_features': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@application.route('/model-info')
def model_info():
    """Display model information and performance metrics"""
    if model is None:
        return "Model not trained yet!"
    
    return render_template('model_info.html',
                         features=FEATURE_NAMES,
                         coefficients=model.coef_,
                         intercept=model.intercept_)

if __name__ == '__main__':
    application.run(debug=True, port=5000)