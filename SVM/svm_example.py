
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('updated_pollution_dataset.csv')

# Encode the target variable (Air Quality)
label_encoder = LabelEncoder()
data['Air Quality'] = label_encoder.fit_transform(data['Air Quality'])

# Define features and target
X = data.drop(columns=['Air Quality'])
y = data['Air Quality']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Save the trained model and scaler
joblib.dump(svm_model, 'svm_air_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Print evaluation results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

