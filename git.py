# Credit Card Fraud Detection
# Simple Version for Online Compiler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample Dataset
# Features:
# [Transaction Amount, Transaction Time]
# Target:
# 0 = Genuine
# 1 = Fraud

X = np.array([
    [100, 10],
    [200, 20],
    [150, 15],
    [1000, 2],
    [1200, 1],
    [1300, 3],
    [110, 18],
    [170, 25],
    [1400, 2],
    [90, 30]
])

Y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 0])

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, Y_train)

# Predictions
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, prediction)

print("Model Accuracy:", accuracy)

print("\n----- Credit Card Fraud Detection -----")

# User Input
amount = float(input("Enter Transaction Amount: "))
time = float(input("Enter Transaction Time: "))

# Prepare Input Data
input_data = np.array([[amount, time]])

# Predict
result = model.predict(input_data)

# Output
if result[0] == 0:
    print("\nTransaction is Genuine")
else:
    print("\nTransaction is Fraudulent")