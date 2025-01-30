from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained model
model = load_model('model.h5', compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Recreate the preprocessing pipeline
numeric_features = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
categorical_features = ["brand", "model", "seller_type", "fuel_type", "transmission_type"]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit the preprocessor with a dataset (replace with actual training data)
training_data = pd.read_csv('C:\\Users\\vedan\\Downloads\\cardekho_dataset.csv\\cardekho_dataset.csv')  # Replace with your training dataset path
 # Replace with your dataset
preprocessor.fit(training_data)

# Flask app setup
app = Flask(__name__)

# Route to render the form
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        input_data = {
            "brand": request.form['brand'],
            "model": request.form['model'],
            "vehicle_age": int(request.form['vehicle_age']),
            "km_driven": int(request.form['km_driven']),
            "seller_type": request.form['seller_type'],
            "fuel_type": request.form['fuel_type'],
            "transmission_type": request.form['transmission_type'],
            "mileage": float(request.form['mileage']),
            "engine": float(request.form['engine']) if request.form['engine'] else None,
            "max_power": float(request.form['max_power']),
            "seats": int(request.form['seats'])
        }

        # Preprocess the input data
        input_df = pd.DataFrame([input_data])
        processed_input = preprocessor.transform(input_df).toarray()

        # Make prediction
        predicted_price = model.predict(processed_input)
        predicted_price = float(predicted_price[0][0])

        # Return the prediction to the user
        return render_template('index.html', prediction=f"Predicted Price: {predicted_price:.2f}")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
