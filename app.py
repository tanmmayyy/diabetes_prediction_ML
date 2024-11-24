# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

app = Flask(__name__)

# Load the trained model and scaler
def load_model():
    try:
        # Load the Random Forest model (we'll save this later)
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        # If model doesn't exist, train it
        return train_and_save_model()

def train_and_save_model():
    # Load and prepare the data
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv', 
                     names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                            'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'])
    
    # Data preprocessing
    columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for column in columns_to_check:
        zero_mask = df[column] == 0
        df.loc[zero_mask, column] = np.nan
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)

    # Split features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    # Save the model and scaler
    joblib.dump(model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return model, scaler

# Load the model and scaler
model, scaler = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Create a dictionary to store the values
        input_data = {}
        for feature in features:
            input_data[feature] = float(request.form[feature])

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'High risk of diabetes' if prediction == 1 else 'Low risk of diabetes'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)