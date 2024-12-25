import joblib
import numpy as np

# Load the model
model = joblib.load('price_prediction_model.pkl')

# Example input data for prediction (replace with actual feature values)
example_input = np.array([[6000,3,0,2000,2.699072554837388,0]])  # Replace with real values

# Predict price
predicted_price = model.predict(example_input)
print("Predicted Price:", predicted_price[0])