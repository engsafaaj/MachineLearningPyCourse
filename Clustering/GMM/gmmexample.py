import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
data_path = 'customer_data.csv'
data = pd.read_csv(data_path)

# Preprocessing: Encode categorical columns and scale numeric data
categorical_columns = ['gender', 'education', 'country']
numeric_columns = ['age', 'income', 'purchase_frequency', 'spending']

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numeric data
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Combine processed data
X = data[categorical_columns + numeric_columns]

# Train Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=8, random_state=42)
gmm.fit(X)

# Save the model and preprocessing tools
with open('gmm_model.pkl', 'wb') as model_file:
    pickle.dump(gmm, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

print("Model training complete. Files saved: gmm_model.pkl, scaler.pkl, label_encoders.pkl")
