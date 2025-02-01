import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Size (sqft)': [1500, 1800, 2400, 3000, 3500],
    'Rooms': [3, 4, 3, 5, 4],
    'Price': [300000, 400000, 500000, 600000, 700000]
}
df = pd.DataFrame(data)

# Split data into training and testing sets
X = df[['Size (sqft)', 'Rooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree regressor
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = reg.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(reg, feature_names=['Size (sqft)', 'Rooms'], filled=True)
plt.title("Decision Tree for Regression")
plt.show()
