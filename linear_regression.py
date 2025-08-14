import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("house_prices.csv")  # CSV should have columns: size, price
X = data["size"].values
y = data["price"].values

# 2. Preprocess (reshape and add bias term)
X = X.reshape(-1, 1)
X = np.c_[np.ones(X.shape[0]), X]  # add column of 1s for bias

# 3. Initialize parameters
theta = np.zeros(X.shape[1])
alpha = 0.0001   # learning rate
epochs = 1000

# 4. Gradient Descent
for _ in range(epochs):
    y_pred = np.dot(X, theta)
    error = y_pred - y
    gradients = (1/len(y)) * np.dot(X.T, error)
    theta -= alpha * gradients

print(f"Final parameters: {theta}")

# 5. Plot results
plt.scatter(data["size"], y, color="blue", label="Data points")
plt.plot(data["size"], np.dot(X, theta), color="red", label="Best fit line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.show()
