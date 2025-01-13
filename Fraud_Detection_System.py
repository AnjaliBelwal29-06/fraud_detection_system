# Importing necessary libraries
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Example dataset structure: ['transaction_id', 'transaaction_amount','location','merchant','age','gender','fraud_label']
data = pd.read_csv("fraud_dataset.csv")  # Replace with your dataset(path)

# Display first few rows of the dataset
print(data.head())

# Step 2: Data Preprocessing
# Remove unnecessary columns (e.g., ID columns)
data = data.drop(columns=['transaction_id'])
#print(data)

# Handle missing values (if any)
#data = data.fillna(data.mean())
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Convert categorical variables into numerical ones (if any)
data['gender'] = data['gender'].map({'M': 0, 'F': 1})
data['merchant'] = data['merchant'].map({'ABC Corp': 0, 'XYZ Inc': 1})
data['location'] = data['location'].map({'New York': 0, 'Chicago': 1,'Los Angeles': 2,'San Francisco':3})

# Split data into features (X) and target (y)
X = data.drop(columns=['fraud_label'])
y = data['fraud_label']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 6: Predict on the test set
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy....
accuracy = model.score(X_test_scaled, y_test)
print(f"\nAccuracy: {accuracy:.2f}")
