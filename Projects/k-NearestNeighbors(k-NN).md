
# k-Nearest Neighbors: Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Dataset Overview](#3-dataset-overview)
4. [Data Preprocessing](#4-data-preprocessing)
5. [The Math Behind k-Nearest Neighbors](#5-the-math-behind-k-nearest-neighbors)
6. [Building and Training the k-NN Model](#6-building-and-training-the-k-nn-model)
7. [Model Evaluation](#7-model-evaluation)
8. [Conclusion](#8-conclusion)
9. [Full Code](#9-full-code)

## 1. Introduction

k-Nearest Neighbors (k-NN) is a simple, non-parametric classification algorithm. It works by finding the 'k' closest data points (neighbors) in the training dataset to a new, unknown data point. Based on the majority class of those neighbors, k-NN makes a classification decision.

In this lesson, we will build a k-NN model using a built-in dataset from Scikit-learn. Specifically, we will use the famous **Iris dataset**, which consists of features describing different species of flowers. Our goal will be to classify new flowers based on their features.

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

## 3. Dataset Overview

The Iris dataset contains 150 samples, each belonging to one of three classes of flowers: Setosa, Versicolour, or Virginica. The features in the dataset include:
- Sepal length
- Sepal width
- Petal length
- Petal width

Each flower class has 50 samples, and we will use these features to classify new samples into one of these three classes.

First, let's load the dataset.

```python
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame for better visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Adding the target (class labels)

# Show the first five rows of the dataset
df.head()
```

## 4. Data Preprocessing

Before building the k-NN model, we need to split the dataset into **training** and **testing** sets. The training set will be used to fit the model, and the testing set will evaluate how well the model performs on unseen data.

```python
# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split data into features (X) and labels (y)
X = iris.data
y = iris.target

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the training and testing sets
print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")
```

## 5. The Math Behind k-Nearest Neighbors

The k-Nearest Neighbors algorithm is based on the principle of **distance calculation** between data points. The algorithm calculates the distance between a new input data point and all other points in the dataset, selects the nearest "k" points, and assigns the class label based on a majority vote.

### Euclidean Distance

The most common distance measure used in k-NN is **Euclidean distance**, which can be calculated as follows:

```
d(p, q) = sqrt((q1 - p1)^2 + (q2 - p2)^2 + ... + (qn - pn)^2)
```

Where:
- **d(p, q)** is the Euclidean distance between points `p` and `q`.
- **p1, p2, ..., pn** are the feature values of point `p`.
- **q1, q2, ..., qn** are the feature values of point `q`.

### Decision Rule

Once the distances are calculated, the k-NN algorithm selects the "k" closest neighbors and determines the class of the new point by **majority voting**. This means the algorithm assigns the class that appears most frequently among the nearest neighbors.

If **k=3**, the algorithm looks for the 3 nearest neighbors and assigns the class based on which class occurs the most out of those 3.

For example:
- If among the 3 neighbors, two points belong to class "1" and one point belongs to class "0," the new data point will be classified as class "1."

## 6. Building and Training the k-NN Model

Now that we have our training and testing sets, we can build the k-NN model. We'll use Scikit-learn's `KNeighborsClassifier` and train the model using the training data.

```python
# Import k-NN classifier
from sklearn.neighbors import KNeighborsClassifier

# Initialize the k-NN classifier with k=3 (3 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn.fit(X_train, y_train)
```

## 7. Model Evaluation

Once the model is trained, it's important to evaluate how well it performs. We can use several metrics, such as accuracy and confusion matrix.

### Predicting on the Test Set

```python
# Make predictions on the test set
y_pred = knn.predict(X_test)
```

### Evaluating Accuracy

```python
# Import accuracy_score from Scikit-learn
from sklearn.metrics import accuracy_score

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

### Confusion Matrix

The confusion matrix is a useful tool to understand the types of errors the model is making.

```python
# Import confusion_matrix from Scikit-learn
from sklearn.metrics import confusion_matrix

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
```

## 8. Conclusion

In this project, we learned how to implement k-Nearest Neighbors (k-NN) using Scikit-learn. We covered:
- Loading and visualizing the Iris dataset.
- Splitting the dataset into training and testing sets.
- Training a k-NN classifier using the training data.
- Evaluating the model's performance using accuracy and a confusion matrix.
- The mathematical principles behind k-NN, including Euclidean distance and majority voting.

k-NN is a simple yet powerful algorithm, and understanding it is important for grasping more complex machine learning models.

## 9. Full Code

Below is the full code for the entire k-NN implementation:

```python
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()

# Convert to a pandas DataFrame for better visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Adding the target (class labels)

# Split data into features (X) and labels (y)
X = iris.data
y = iris.target

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier with k=3 (3 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
```

