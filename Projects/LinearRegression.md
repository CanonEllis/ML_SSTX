
# Linear Regression: Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Code Walkthrough](#3-code-walkthrough)
4. [Mean Squared Error (MSE) Explanation](#4-mean-squared-error-mse-explanation)
5. [R-squared Explanation](#5-r-squared-explanation)
6. [Jupyter Notebook with Code and Comments](#6-jupyter-notebook-with-code-and-comments)
7. [Conclusion](#7-conclusion)

## 1. Introduction
Linear regression is a basic statistical technique used to model the relationship between one or more independent variables (features) and a dependent variable (target). It assumes a linear relationship between the input variables and the output variable. In the case of simple linear regression (with one feature), the function takes the following form:

```
y = beta_0 + beta_1 * x
```

Where:
- **y** is the predicted value (dependent variable).
- **beta_0** is the intercept (constant).
- **beta_1** is the slope or coefficient for the input feature.
- **x** is the input feature (independent variable).

In the case of multiple linear regression, where there are multiple features, the equation becomes:

```
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_n * x_n
```

## 2. Prerequisites

To run this project, you will need to install the following Python libraries:
- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`

You can install them using the following command:

```bash
pip install numpy scikit-learn matplotlib pandas
```

## 3. Code Walkthrough

### Linear Regression Code

This code demonstrates how to implement linear regression using Scikit-learn's `LinearRegression` class.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some random linear data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Plot the results
plt.scatter(X, y, color="gray", label="True data")
plt.plot(X_test, y_pred_test, color="red", label="Linear regression")
plt.xlabel("Input Feature (X)")
plt.ylabel("Output (y)")
plt.title("Linear Regression")
plt.legend()
plt.show()

# Calculate Mean Squared Error and R-squared for model evaluation
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Print the evaluation metrics
print(f"Train Mean Squared Error (MSE): {mse_train:.3f}")
print(f"Test Mean Squared Error (MSE): {mse_test:.3f}")
print(f"Train R-squared: {r2_train:.3f}")
print(f"Test R-squared: {r2_test:.3f}")
```

### Explanation:
- **LinearRegression**: This class is used to fit a linear regression model. It minimizes the residual sum of squares between the observed and predicted values.
- **Mean Squared Error**: Measures the average squared difference between the actual and predicted values.
- **R-squared**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

## 4. Mean Squared Error (MSE) Explanation

The mean squared error (MSE) is a metric that calculates the average squared difference between the actual values (y_i) and the predicted values (y_hat_i):

```
MSE = (1/n) * sum((y_i - y_hat_i)^2)
```

Where:
- **n** is the number of samples.
- **y_i** is the actual value.
- **y_hat_i** is the predicted value.

Lower MSE indicates a better fit.

## 5. R-squared Explanation

R-squared (R^2) is a metric that indicates how well the independent variables explain the variance in the dependent variable. It is defined as:

```
R^2 = 1 - (sum(y_i - y_hat_i)^2 / sum(y_i - y_mean)^2)
```

Where:
- **y_i** is the actual value.
- **y_hat_i** is the predicted value.
- **y_mean** is the mean of the actual values.

Higher R^2 values indicate that the model explains more variance.

## 6. Jupyter Notebook with Code and Comments

Below is the code for linear regression with ample comments to help you understand each step.

```python
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some random linear data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate Mean Squared Error and R-squared for model evaluation
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Print the evaluation metrics
print(f"Train Mean Squared Error (MSE): {mse_train:.3f}")
print(f"Test Mean Squared Error (MSE): {mse_test:.3f}")
print(f"Train R-squared: {r2_train:.3f}")
print(f"Test R-squared: {r2_test:.3f}")
```

## 7. Conclusion

In this project, you learned:
- How to implement linear regression using Scikit-learn.
- How to evaluate the model using metrics like Mean Squared Error (MSE) and R-squared.
- The significance of R^2 and MSE in evaluating the performance of a regression model.
