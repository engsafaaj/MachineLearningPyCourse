import pandas as pd
import pickle

# Load the trained model and preprocessing tools
with open('gmm_model.pkl', 'rb') as model_file:
    gmm = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Function to preprocess new input
def preprocess_input(data, label_encoders, scaler):
    categorical_columns = ['gender', 'education', 'country']
    numeric_columns = ['age', 'income', 'purchase_frequency', 'spending']

    for col in categorical_columns:
        le = label_encoders[col]
        data[col] = le.transform(data[col])

    data[numeric_columns] = scaler.transform(data[numeric_columns])
    return data[categorical_columns + numeric_columns]

# Example new input data
new_data = pd.DataFrame({
    'gender': ['Female'],
    'education': ['Bachelor'],
    'country': ['Iraq'],
    'age': [25],
    'income': [30000],
    'purchase_frequency': [0.8],
    'spending': [5000]
})

# Preprocess the new data
processed_data = preprocess_input(new_data, label_encoders, scaler)

# Predict using the GMM model
predictions = gmm.predict(processed_data)
print("Predicted cluster:", predictions[0])
