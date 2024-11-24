# This would be a simple but inaccurate way
def predict_diabetes_with_if_else(glucose, bmi):
    if glucose > 140 and bmi > 30:
        return "High Risk"
    else:
        return "Low Risk"
        
        






# Demonstration of how the model actually works

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 1. First, we train the model with real data
def train_model():
    # Load the diabetes dataset
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv', 
                     names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Train the model
model, scaler = train_model()

# 2. Let's make predictions with same input using both methods
def compare_predictions():
    # Sample input
    sample_input = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    # Method 1: Simple if-else (oversimplified)
    def if_else_prediction(data):
        if data['Glucose'] > 140 and data['BMI'] > 30:
            return 1  # High Risk
        return 0  # Low Risk
    
    # Method 2: ML Model prediction (actual method used)
    def model_prediction(data):
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Scale the input using same scaler used in training
        input_scaled = scaler.transform(input_df)
        
        # Get prediction and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        return prediction, probability
    
    # Compare results
    if_else_result = if_else_prediction(sample_input)
    model_result, probability = model_prediction(sample_input)
    
    print("Comparison of Methods:")
    print("\n1. Simple If-Else Prediction:")
    print(f"Result: {'High Risk' if if_else_result == 1 else 'Low Risk'}")
    print("\n2. Machine Learning Model Prediction:")
    print(f"Result: {'High Risk' if model_result == 1 else 'Low Risk'}")
    print(f"Probability: {probability:.2%}")
    
    # Show how model makes complex decisions
    print("\nModel's Important Features:")
    feature_importance = pd.DataFrame({
        'Feature': list(sample_input.keys()),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)

# Run comparison
compare_predictions()