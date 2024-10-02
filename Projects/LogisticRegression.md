
# Logistic Regression: Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Code Walkthrough](#3-code-walkthrough)
4. [Log Loss Explanation](#4-log-loss-explanation)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Jupyter Notebook with Code and Comments](#6-jupyter-notebook-with-code-and-comments)
7. [Conclusion](#7-conclusion)

## 1. Introduction
This project demonstrates how to implement logistic regression using Scikit-learn and evaluate the model using key metrics such as accuracy, precision, recall, F1-score, and log loss. In addition, it provides explanations for the mathematical concepts behind these metrics.

The notebook included in this file provides a hands-on demonstration of logistic regression for binary classification.

## 2. Prerequisites

To run this project, you will need to install the following Python libraries:
- `numpy`
- `scikit-learn`
- `matplotlib` (if you'd like to visualize the loss)

You can install them using the following command:

```bash
pip install numpy scikit-learn matplotlib
```

## 3. Code Walkthrough

### Logistic Regression

Logistic regression is used for binary classification tasks where the goal is to predict whether an input belongs to one of two classes. The model estimates the probability that a given input belongs to class 1 and classifies it accordingly.


![Alt Text](./images/logreg_graph.png)

#### Sigmoid Function

The logistic regression model is based on the sigmoid function, which maps any real-valued number into a probability between 0 and 1.

The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- \( z \) is the linear combination of input features and model coefficients.

The sigmoid function outputs a probability value between 0 and 1, which is then used to classify the input into one of two categories. If the predicted probability is greater than or equal to 0.5, the model classifies the input as class 1. Otherwise, it classifies it as class 0.

### Logistic Regression Code

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Load dataset (Iris dataset example)
from sklearn.datasets import load_iris
iris = load_iris()

# Use only 2 classes for binary classification
X = iris.data
y = (iris.target == 2).astype(int)  # Binary classification: Is it Iris-Virginica or not?

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions and calculate probabilities
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)

# Print metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Log Loss: {loss:.2f}")
```

## 4. Log Loss Explanation

Log loss, also known as logarithmic loss or logistic loss, is the evaluation metric used for logistic regression. It measures how well the model's predicted probabilities match the actual class labels. Log loss penalizes incorrect predictions, especially if the model is confident about the wrong prediction.

The formula for log loss is:

$$
    {Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right]
$$

Where:
- \( N \) is the number of samples.
- \( y_i \) is the actual label (0 or 1) for sample \( i \).
- \( p_i \) is the predicted probability that sample \( i \) belongs to class 1.

Lower log loss indicates a better model.

## 5. Evaluation Metrics

In addition to log loss, there are other important metrics used to evaluate classification models:

### Accuracy

Accuracy measures the proportion of correct predictions (both true positives and true negatives) out of the total predictions.

$$
    {Accuracy} = \frac{	{True Positives} + 	{True Negatives}}{	{Total Samples}}
$$

### Precision

Precision is the proportion of true positive predictions out of all positive predictions (both true positives and false positives).

$$
    {Precision} = \frac{	{True Positives}}{	{True Positives} + 	{False Positives}}
$$

### Recall

Recall measures the proportion of actual positive samples that were correctly predicted.

$$
    {Recall} = \frac{	{True Positives}}{	{True Positives} + 	{False Negatives}}
$$

### F1-Score

The F1-score is the harmonic mean of precision and recall. It balances both metrics.

$$
    {F1-Score} = 2 \times \frac{	{Precision} \times 	{Recall}}{	{Precision} + 	{Recall}}
$$

## 6. Jupyter Notebook with Code and Comments

Below is the code for logistic regression with ample comments to help you understand each step.

```python
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Load the Iris dataset
from sklearn.datasets import load_iris
iris = load_iris()

# We will use only 2 classes (binary classification)
X = iris.data
y = (iris.target == 2).astype(int)  # Convert to binary classification problem

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
logreg = LogisticRegression()

# Train the Logistic Regression model using the training data
logreg.fit(X_train, y_train)

# Make predictions using the test data
y_pred = logreg.predict(X_test)

# Calculate the probabilities for each class
y_prob = logreg.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Iris-Virginica)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)

# Print the evaluation metrics to see how well the model performed
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Log Loss: {loss:.2f}")
```

## 7. Conclusion

In this project, you learned:
- How to implement logistic regression for binary classification using Scikit-learn.
- The significance of the sigmoid function in logistic regression.
- How log loss works and why it's important.
- Key evaluation metrics like accuracy, precision, recall, and F1-score.

Understanding these concepts is essential for building and evaluating machine learning models, especially when working with classification problems.
