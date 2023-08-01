from flask import Flask, request, jsonify, render_template
import boto3
import pandas as pd
import numpy as np
import sagemaker
from sagemaker.predictor import csv_serializer

app = Flask(__name__)

# Replace with the actual endpoint name of your deployed XGBoost model
endpoint_name = 'xgboost-2023-07-31-17-04-28-845'

# Initialize the SageMaker predictor for the endpoint
sess = boto3.Session()
sagemaker_predictor = sagemaker.predictor.RealTimePredictor(endpoint_name, sagemaker_session=sess)
sagemaker_predictor.content_type = 'text/csv'
sagemaker_predictor.serializer = csv_serializer

# Read the diabetes data from the CSV file
data = pd.read_csv('/Users/elenayun/Desktop/DG/diabetes_prediction_dataset.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.form.to_dict()
        gender = data['gender']
        age = float(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        blood_glucose_level = float(data['blood_glucose_level'])
        
        # Convert the features to a comma-separated string for prediction
        features_str = f'{gender},{age},{hypertension},{heart_disease},{blood_glucose_level},0'
        
        # Make the prediction using the deployed model
        prediction = sagemaker_predictor.predict(features_str).decode('utf-8')
        
        # Convert the prediction to an integer
        prediction = int(float(prediction))
        
        # Map the integer prediction to the target variable (diabetes)
        target_variable = {0: 'No Diabetes', 1: 'Diabetes'}
        predicted_diabetes = target_variable[prediction]
        
        # Return the prediction as a response
        return render_template('index.html', prediction_text=f'Diabetes: {predicted_diabetes}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
