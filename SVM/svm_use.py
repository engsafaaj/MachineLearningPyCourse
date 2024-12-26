
import joblib

# Load the saved model and scaler
model = joblib.load('svm_air_quality_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example input data (replace with actual data)
input_data = [[29.8, 59.1, 5.2, 17.9, 18.9, 9.2, 1.72, 6.3, 319]]

# Preprocess the input data
input_data_scaled = scaler.transform(input_data)

# Make a prediction
prediction = model.predict(input_data_scaled)

# Decode the prediction
air_quality_classes = {0: "Good", 1: "Hazardous", 2: "Moderate", 3: "Poor"}
print(f"Predicted air quality class: {air_quality_classes[prediction[0]]}")
