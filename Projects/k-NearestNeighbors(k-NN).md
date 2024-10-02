# Support Vector Machines (SVM): Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Dataset Overview](#3-dataset-overview)
4. [Data Preprocessing](#4-data-preprocessing)
5. [The Math Behind SVM](#5-the-math-behind-svm)
6. [Building and Training the SVM Model](#6-building-and-training-the-svm-model)
7. [Model Evaluation](#7-model-evaluation)
8. [Visualization](#8-visualization)
9. [Conclusion](#9-conclusion)
10. [Full Code](#10-full-code)

## 1. Introduction

Support Vector Machines (SVM) are supervised learning models used for classification and regression analysis. They are particularly effective in high-dimensional spaces and are useful when the number of dimensions is greater than the number of samples.

In this lesson, we will build an SVM model using a built-in dataset from Scikit-learn. Specifically, we will use the **Iris dataset**, which consists of features describing different species of flowers. Our goal will be to classify new flowers based on their features.

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

Before building the SVM model, we need to split the dataset into **training** and **testing** sets. The training set will be used to fit the model, and the testing set will evaluate how well the model performs on unseen data.

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

## 5. The Math Behind SVM

SVM works by finding a hyperplane that best divides a dataset into two classes. The best hyperplane is the one that maximizes the **margin** between the classes. In cases where data isn't linearly separable, SVM can use kernel functions to project data into higher dimensions.

### Hyperplane

A **hyperplane** is a decision boundary that separates the data points of different classes. For instance:
- In 2D space, a hyperplane is a line.
- In 3D space, a hyperplane is a plane.
- In higher dimensions, it becomes a generalized decision boundary.

SVM aims to find the hyperplane that has the largest margin, meaning the greatest distance between the hyperplane and the nearest data points from each class (these points are called **support vectors**).

### SVM Objective

The objective of SVM is to find a hyperplane defined by the equation:

```
w * x - b = 0
```

Where:
- **w** is the weight vector.
- **x** is the input vector.
- **b** is the bias term.

SVM maximizes the margin between the two classes, which is the distance between the closest points in the two classes (called **support vectors**) and the hyperplane.

### Hinge Loss

The hinge loss function is used to ensure a large margin and is given by:

```
L(w, b) = sum(max(0, 1 - y_i(w * x_i - b)))
```

Where:
- **x_i** is the input vector for the i-th sample.
- **y_i** is the class label (+1 or -1) for the i-th sample.

## 6. Building and Training the SVM Model

Now that we have our training and testing sets, we can build the SVM model. We'll use Scikit-learn's `SVC` (Support Vector Classifier) and train the model using the training data.

```python
# Import the SVM classifier
from sklearn.svm import SVC

# Initialize the SVM classifier
svm = SVC(kernel='linear', random_state=42)

# Train the model on the training data
svm.fit(X_train, y_train)
```

### Explanation:
- We import `SVC` from Scikit-learn and initialize the classifier using a linear kernel.
- The model is then trained using the `.fit()` method, which takes in the training data (`X_train`) and the corresponding labels (`y_train`).

## 7. Model Evaluation

Once the model is trained, it's important to evaluate how well it performs. We can use several metrics, such as accuracy and confusion matrix.

### Predicting on the Test Set

```python
# Make predictions on the test set
y_pred = svm.predict(X_test)
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

### Explanation:
- A confusion matrix provides insight into the performance of the classification model by showing the number of true positives, false positives, true negatives, and false negatives. It helps in understanding which classes are being misclassified.

## 8. Visualization

To better understand how SVM works, we can visualize the decision boundary that the SVM creates, along with the support vectors. We will plot this for a subset of the features (2D plot).

```python
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Create a simple synthetic dataset for visualization
X, y = make_classification(n_classes=2, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Generate a mesh grid for the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict the class labels for each point in the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create a color map for the plot
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#0000FF']

# Plot the decision boundary and the training points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=100)
plt.title("K-Nearest Neighbors (KNN) Visualization")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
```

Here is a visualization of the K-Nearest Neighbors (KNN) classification algorithm.

The background color represents the decision boundary, indicating how the classifier assigns points to either class based on the training data.  
The red and blue points are the data points from the two classes in the dataset.  
KNN classifies a new point by looking at the \( k \) nearest neighbors (in this case, 5 neighbors) and assigning the class that appears most frequently among those neighbors.  
This visual shows how KNN forms decision boundaries based on the nearest points in feature space.


### Explanation:
- We use only two features from the dataset to visualize the decision boundary in a 2D plot.
- A meshgrid is created to map the decision boundary using the trained SVM model.
- The plot shows the decision boundary, the data points, and highlights the support vectors.

## 9. Conclusion

In this project, we learned how to implement Support Vector Machines (SVM) using Scikit-learn. We covered:
- Loading and visualizing the Iris dataset.
- Splitting the dataset into training and testing sets.
- Training an SVM classifier using the training data.
- Evaluating the model's performance using accuracy and a confusion matrix.
- Visualizing the decision boundary and understanding the concept of support vectors.

Support Vector Machines are powerful algorithms, especially in high-dimensional spaces, and are commonly used in both classification and regression tasks.

## 10. Full Code

Below is the full code for the entire SVM implementation:

```python
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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

# Initialize the SVM classifier
svm = SVC(kernel='linear', random_state=42)

# Train the model on the training data
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualization for 2D Decision Boundary
X_vis = X_train[:, :2]
y_vis = y_train

# Train the SVM model on the two features
svm_vis = SVC(kernel='linear', random_state=42)
svm_vis.fit(X_vis, y_vis)

# Create a mesh to plot the decision boundary
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Plot decision boundary
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot the points
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=plt.cm.coolwarm, edgecolors='k')

# Highlight the support vectors
plt.scatter(svm_vis.support_vectors_[:, 0], svm_vis.support_vectors_[:, 1], facecolors='none', s=100, edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary with Support Vectors')
plt.show()
```
