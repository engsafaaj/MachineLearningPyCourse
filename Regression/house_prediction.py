# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('house_price_regression_dataset.csv')

# Prepare data
X = data[['Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']]  # Feature columns
y = data['House_Price']  # Target column

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Save the model
joblib.dump(model, 'price_prediction_model.pkl')
# Draw the line of best fit (for one feature example: 'Lot_Size')
plt.scatter(X_test['Lot_Size'], y_test, color='blue', label='Actual Prices')
plt.scatter(X_test['Lot_Size'], y_pred, color='red', label='Predicted Prices')
plt.plot(X_test['Lot_Size'], y_pred, color='green', label='Best Fit Line')
plt.xlabel('Lot_Size')
plt.ylabel('House_Price')
plt.title('Linear Regression Fit: Lot_Size vs House_Price')
plt.legend()
plt.show()
