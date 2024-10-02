# Decision Trees: Concepts, Code, and Model Evaluation

## Table of Contents
1. [Introduction](#1-introduction)
2. [Prerequisites](#2-prerequisites)
3. [Dataset Overview](#3-dataset-overview)
4. [Data Preprocessing](#4-data-preprocessing)
5. [The Math Behind Decision Trees](#5-the-math-behind-decision-trees)
6. [Building and Training the Decision Tree Model](#6-building-and-training-the-decision-tree-model)
7. [Model Evaluation](#7-model-evaluation)
8. [Conclusion](#8-conclusion)
9. [Full Code](#9-full-code)

## 1. Introduction

Decision Trees are a powerful classification algorithm that splits the data into branches based on feature values. The tree structure consists of **nodes** where decisions are made based on the features, and **leaf nodes** where the final class label is assigned.

In this lesson, we will build a Decision Tree model using a built-in dataset from Scikit-learn. Specifically, we will use the famous **Iris dataset**, which consists of features describing different species of flowers. Our goal will be to classify new flowers based on their features.

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

Before building the Decision Tree model, we need to split the dataset into **training** and **testing** sets. The training set will be used to fit the model, and the testing set will evaluate how well the model performs on unseen data.

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

## 5. The Math Behind Decision Trees

Decision Trees use a recursive splitting process to build a tree where the internal nodes represent decisions based on features. The splitting is typically based on **information gain** or **Gini impurity**.

### Information Gain

Information gain measures how much "purity" is achieved after splitting the data on a specific feature. It is based on **entropy**, which measures the uncertainty in the data.

#### Entropy

The formula for entropy is:

```
H(S) = -sum(p_i * log2(p_i))
```

Where:
- **H(S)** is the entropy of the set `S`.
- **p_i** is the proportion of examples in class `i`.

If all examples belong to one class, the entropy is zero (pure set).

#### Information Gain

Information gain is the reduction in entropy after splitting the data:

```
IG(S, A) = H(S) - sum((|S_v| / |S|) * H(S_v))
```

Where:
- **S** is the set of examples.
- **A** is the feature being split.
- **S_v** is the subset of `S` where feature `A` has value `v`.

### Gini Impurity

Another common metric is **Gini impurity**, which measures the probability that a randomly chosen example would be incorrectly classified.

The formula for Gini impurity is:

```
Gini(S) = 1 - sum(p_i^2)
```

Where **p_i** is the proportion of examples in class `i`.

## 6. Building and Training the Decision Tree Model

Now that we have our training and testing sets, we can build the Decision Tree model. We'll use Scikit-learn's `DecisionTreeClassifier` and train the model using the training data.

```python
# Import Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree classifier
tree = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
tree.fit(X_train, y_train)
```

### Explanation:
- We import `DecisionTreeClassifier` from Scikit-learn and initialize the classifier.
- The model is then trained using the `.fit()` method, which takes in the training data (`X_train`) and the corresponding labels (`y_train`).

## 7. Model Evaluation

Once the model is trained, it's important to evaluate how well it performs. We can use several metrics, such as accuracy and confusion matrix.

### Predicting on the Test Set

```python
# Make predictions on the test set
y_pred = tree.predict(X_test)
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

## 8. Conclusion

In this project, we learned how to implement Decision Trees using Scikit-learn. We covered:
- Loading and visualizing the Iris dataset.
- Splitting the dataset into training and testing sets.
- Training a Decision Tree classifier using the training data.
- Evaluating the model's performance using accuracy and a confusion matrix.
- The mathematical principles behind Decision Trees, including entropy, information gain, and Gini impurity.

Decision Trees are interpretable and powerful algorithms, and understanding them is important for grasping more complex machine learning models.

## 9. Full Code

Below is the full code for the entire Decision Tree implementation:

```python
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Initialize the Decision Tree classifier
tree = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
```
