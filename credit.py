import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv("creditcard.csv")

# Display Dataset Information
print("First 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nChecking Missing Values:")
print(data.isnull().sum())

# Count Genuine and Fraud Transactions
print("\nTransaction Counts:")
print(data['Class'].value_counts())

# Separate Features and Target
X = data.drop(columns='Class', axis=1)
Y = data['Class']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, Y_train)

# Training Prediction
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)

print("\nTraining Accuracy:", training_accuracy)

# Testing Prediction
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Testing Accuracy:", test_accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))

# Classification Report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))

# Custom Prediction System
print("\n----- Fraud Detection System -----")

# Taking sample input
time = float(input("Enter Time: "))
amount = float(input("Enter Amount: "))

# Create dummy V1 to V28 values
# In real projects these come from dataset features
v_values = []

for i in range(1, 29):
    value = float(input(f"Enter V{i}: "))
    v_values.append(value)

# Create Input Array
input_data = [time] + v_values + [amount]

# Convert to NumPy Array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape Input
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Scale Input
std_data = scaler.transform(input_data_reshaped)

# Prediction
prediction = model.predict(std_data)

print("\nPrediction Result:")

if prediction[0] == 0:
    print("The transaction is Genuine")
else:
    print("The transaction is Fraudulent")
