
# Polynomial Regression: Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Code Walkthrough](#3-code-walkthrough)
4. [Mean Squared Error (MSE) Explanation](#4-mean-squared-error-mse-explanation)
5. [R-squared Explanation](#5-r-squared-explanation)
6. [Jupyter Notebook with Code and Comments](#6-jupyter-notebook-with-code-and-comments)
7. [Conclusion](#7-conclusion)

## 1. Introduction
Polynomial regression is an extension of linear regression, which models the relationship between the independent variable(s) and the dependent variable as an \(n\)-degree polynomial. It is used when the relationship between variables is nonlinear.

While linear regression fits a straight line (first-degree polynomial), polynomial regression fits a curve that can capture more complex relationships. The polynomial function can take the following form:

$$
y = &beta;_0 + &beta;_1 x + &beta;_2 x^2 + &dots; + &beta;_n x^n
$$

Where:
- **y** is the predicted value.
- **&beta;_0, &beta;_1, &dots;, &beta;_n** are the model coefficients.
- **x** is the input feature (or independent variable).
- **n** is the degree of the polynomial.

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

### Polynomial Regression Code

This code demonstrates how to implement polynomial regression using Scikit-learn's `PolynomialFeatures` to generate polynomial features and `LinearRegression` to fit the model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some random nonlinear data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.2  # Nonlinear relationship with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the input features to polynomial features (degree 3 in this case)
polynomial_features = PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.transform(X_test)

# Fit polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

# Plot the results
plt.scatter(X, y, color="gray", label="True data")
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_range_poly = polynomial_features.transform(X_range)
plt.plot(X_range, model.predict(X_range_poly), color="red", label="Polynomial regression (degree 3)")
plt.xlabel("Input Feature (X)")
plt.ylabel("Output (y)")
plt.title("Polynomial Regression (degree 3)")
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
- **PolynomialFeatures**: Used to transform the original feature into higher-degree polynomial features. In this case, we use degree 3.
- **LinearRegression**: Once the features are transformed, the linear regression model fits the polynomial regression.
- **Mean Squared Error**: Measures the average squared difference between the actual and predicted values.
- **R-squared**: Measures the proportion of the variance in the dependent variable that is predictable from the independent variables.

## 4. Mean Squared Error (MSE) Explanation

The mean squared error (MSE) is a metric that calculates the average squared difference between the actual values **y<sub>i</sub>** and the predicted values **&hat;y<sub>i</sub>**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- **n** is the number of samples.
- **y<sub>i</sub>** is the actual value.
- **&hat;y<sub>i</sub>** is the predicted value.

Lower MSE indicates a better fit.

## 5. R-squared Explanation

R-squared (**R<sup>2</sup>**) is a metric that indicates how well the independent variables explain the variance in the dependent variable. It is defined as:

$$
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Where:
- **y<sub>i</sub>** is the actual value.
- **&hat;y<sub>i</sub>** is the predicted value.
- **&bar;y** is the mean of the actual values.

Higher **R<sup>2</sup>** values indicate that the model explains more variance.

## 6. Jupyter Notebook with Code and Comments

Below is the code for polynomial regression with ample comments to help you understand each step.

```python
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate some random nonlinear data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.2  # Nonlinear relationship with noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the input features to polynomial features (degree 3 in this case)
polynomial_features = PolynomialFeatures(degree=3)
X_train_poly = polynomial_features.fit_transform(X_train)
X_test_poly = polynomial_features.transform(X_test)

# Fit polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Make predictions
y_pred_train = model.predict(X_train_poly)
y_pred_test = model.predict(X_test_poly)

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
- How to implement polynomial regression for nonlinear data using Scikit-learn.
- How to use the `PolynomialFeatures` class to generate polynomial features.
- How to evaluate the model using metrics like Mean Squared Error (MSE) and R-squared.
- The significance of **R<sup>2</sup>** and MSE in evaluating the performance of a regression model.
