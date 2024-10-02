
# k-Nearest Neighbors: Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Dataset Overview](#3-dataset-overview)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Building and Training the k-NN Model](#5-building-and-training-the-k-nn-model)
6. [Model Evaluation](#6-model-evaluation)
7. [Conclusion](#7-conclusion)

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

### Explanation:
- We use Scikit-learn's `load_iris` function to load the Iris dataset.
- The data is converted into a Pandas DataFrame for easier visualization.
- We print the first five rows of the dataset to familiarize ourselves with the features.

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

### Explanation:
- We split the dataset into features (`X`) and labels (`y`). The features are the measurements (e.g., sepal length, petal width), and the labels are the classes (Setosa, Versicolour, Virginica).
- The dataset is then split into training (80%) and testing (20%) sets using `train_test_split`. The training set is used to train the model, and the testing set is used to evaluate how well the model generalizes to new data.

## 5. Building and Training the k-NN Model

Now that we have our training and testing sets, we can build the k-NN model. We'll use Scikit-learn's `KNeighborsClassifier` and train the model using the training data.

```python
# Import k-NN classifier
from sklearn.neighbors import KNeighborsClassifier

# Initialize the k-NN classifier with k=3 (3 nearest neighbors)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
knn.fit(X_train, y_train)
```

### Explanation:
- We import `KNeighborsClassifier` from Scikit-learn and initialize the classifier with `k=3`. This means the model will look at the 3 nearest neighbors for classification.
- The model is then trained using the `.fit()` method, which takes in the training data (`X_train`) and the corresponding labels (`y_train`).

## 6. Model Evaluation

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

### Explanation:
- We use the `predict()` method to make predictions on the test set (`X_test`).
- To evaluate how well the model performs, we calculate the accuracy using the `accuracy_score` function, which compares the true labels (`y_test`) with the predicted labels (`y_pred`).

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

### Explanation:
- A confusion matrix provides insight into the performance of the classification model by showing the number of true positives, false positives, true negatives, and false negatives. It helps in understanding which classes are being misclassified.

## 7. Conclusion

In this project, we learned how to implement k-Nearest Neighbors (k-NN) using Scikit-learn. We covered:
- Loading and visualizing the Iris dataset.
- Splitting the dataset into training and testing sets.
- Training a k-NN classifier using the training data.
- Evaluating the model's performance using accuracy and a confusion matrix.

k-NN is a simple yet powerful algorithm, and understanding it is important for grasping more complex machine learning models.
