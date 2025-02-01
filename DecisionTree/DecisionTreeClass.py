import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Age': [25, 30, 45, 35, 40, 50, 23],
    'Income': [40000, 50000, 60000, 80000, 100000, 150000, 20000],
    'Purchased': [0, 0, 1, 1, 1, 1, 0]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

# Split data into training and testing sets
X = df[['Age', 'Income']]
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=['Age', 'Income'], class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Classification")
plt.show()
