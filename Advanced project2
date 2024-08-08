# AI-Self-Driving-Cars
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset (replace with your actual data loading method)
data = pd.read_csv("your_dataset.csv")

# Preprocess the data
# ...

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("target_variable", axis=1), data["target_variable"], test_size=0.2)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse}")
