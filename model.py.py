# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset (make sure train.csv is in same folder)
df = pd.read_csv("train.csv")

# Select required features
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

# Drop missing values (if any)
df = df.dropna()

# Define X (features) and y (target)
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Print coefficients
print("\nModel Coefficients:")
print("Square Footage:", model.coef_[0])
print("Bedrooms:", model.coef_[1])
print("Bathrooms:", model.coef_[2])
print("Intercept:", model.intercept_)

# BONUS: Visualization (Actual vs Predicted)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# Test with your own input
sample = np.array([[2000, 3, 2]])  # 2000 sqft, 3 bed, 2 bath
predicted_price = model.predict(sample)
print("\nPredicted Price for 2000 sqft, 3 bed, 2 bath:", predicted_price[0])