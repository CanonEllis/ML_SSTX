# Introduction to Deep Learning

## Table of Contents
1. [Introduction](#1-introduction)
2. [What is Deep Learning?](#2-what-is-deep-learning)
3. [Neural Networks](#3-neural-networks)
4. [How Neural Networks Learn](#4-how-neural-networks-learn)
5. [Activation Functions](#5-activation-functions)
6. [Loss Function and Optimization](#6-loss-function-and-optimization)
7. [Popular Architectures](#7-popular-architectures)
8. [Glossary](#8-glossary)

## 1. Introduction

Deep learning is a subset of machine learning where models, known as neural networks, are composed of multiple layers that process data. Each layer of a neural network extracts higher-level features from the raw input. Deep learning has become popular due to its success in image recognition, natural language processing, and other complex tasks.

In this introduction, we will explore key concepts in deep learning, such as neural networks, activation functions, and how models learn through optimization. We will also look at some popular deep learning architectures.

## 2. What is Deep Learning?

**Deep Learning** is a type of machine learning that uses models called **neural networks**. What distinguishes deep learning from traditional machine learning is the **depth** of the model, i.e., the number of layers in the network. Deep learning models are particularly good at tasks like image classification, speech recognition, and language translation.

### Key Concepts in Deep Learning:
- **Neural Networks**: Composed of layers of artificial neurons that pass data between them.
- **Backpropagation**: A method used to adjust the weights of the neural network to minimize the error.
- **Supervised Learning**: A type of learning where the model is trained on labeled data.

## 3. Neural Networks

A **Neural Network** is a model that mimics the human brain. It is composed of **neurons**, or nodes, connected in layers. The **input layer** receives data, **hidden layers** process the data, and the **output layer** produces the result.

### Structure of a Neural Network:
- **Input Layer**: The first layer that receives the input data.
- **Hidden Layers**: Layers that process inputs, transforming them into more abstract features. These layers are called "hidden" because they are not visible to the outside.
- **Output Layer**: The final layer that produces the prediction or classification.

### Example of a Neural Network

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a simple neural network
model = Sequential()

# Add layers: Input layer, 2 hidden layers, and an output layer
model.add(Dense(units=64, activation='relu', input_shape=(100,)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### Explanation:
- The **Sequential** model is a linear stack of layers.
- **Dense** layers are fully connected layers, meaning every node in the previous layer connects to every node in the current layer.
- The input shape is (100,), meaning the network expects 100 features as input.
- The final layer has 10 units with a **softmax** activation function, useful for classification problems.

## 4. How Neural Networks Learn

Neural networks learn by adjusting the weights of the connections between neurons. The goal is to minimize the difference between the predicted output and the actual output, which is measured by a **loss function**.

### Steps in Learning:
1. **Forward Pass**: Input data is passed through the network to make predictions.
2. **Loss Calculation**: The error or difference between predicted output and actual output is calculated using a loss function.
3. **Backpropagation**: The error is propagated back through the network to adjust the weights using **gradient descent**.
4. **Weight Update**: The weights are updated to reduce the error. This process is repeated until the error is minimized.

## 5. Activation Functions

**Activation Functions** determine the output of a neuron in a neural network. They introduce non-linearity, allowing the network to learn more complex patterns.

### Common Activation Functions:
- **Sigmoid**: Maps any input to a value between 0 and 1. Useful for binary classification.
  ```
  Sigmoid(x) = 1 / (1 + exp(-x))
  ```
- **ReLU (Rectified Linear Unit)**: Outputs the input if positive, otherwise outputs zero. Popular due to its simplicity and effectiveness.
  ```
  ReLU(x) = max(0, x)
  ```
- **Softmax**: Converts raw class scores into probabilities, often used in multi-class classification.
  ```
  Softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
  ```

### Example Activation Function in Code:

```python
model.add(Dense(units=64, activation='relu'))  # ReLU activation function
```

## 6. Loss Function and Optimization

### Loss Function

The **loss function** measures how well the model's predictions match the actual labels. In deep learning, common loss functions include:

- **Mean Squared Error (MSE)**: Often used for regression tasks. It calculates the average of the squared differences between predicted and actual values.
  ```
  MSE = (1/n) * sum((y_true - y_pred)^2)
  ```
- **Cross-Entropy**: Used for classification problems. It measures the difference between two probability distributions.
  ```
  Cross-Entropy = -sum(y_true * log(y_pred))
  ```

### Optimization Algorithm

**Optimization algorithms** are used to minimize the loss function. The most commonly used algorithm in deep learning is **Gradient Descent**.

### Example of Model Compilation with a Loss Function and Optimizer

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

In this example:
- **Adam**: A popular optimization algorithm that combines the benefits of both momentum and adaptive learning rate methods.
- **Categorical Cross-Entropy**: A loss function used for multi-class classification.

## 7. Popular Architectures

There are several popular architectures in deep learning, each designed for different tasks:

- **Convolutional Neural Networks (CNNs)**: Primarily used for image recognition tasks.
- **Recurrent Neural Networks (RNNs)**: Effective for sequence prediction tasks such as time series or natural language processing.
- **Generative Adversarial Networks (GANs)**: Used for generating new data that resembles training data.

Each architecture is specialized for certain tasks, and understanding when to use each one is crucial for building effective deep learning models.

## 8. Glossary

### Activation Function
- **Definition**: A mathematical function that determines the output of a neuron given an input. It introduces non-linearity into the model.

### Backpropagation
- **Definition**: A method used to calculate the gradient of the loss function with respect to each weight in the network, enabling the model to update the weights.

### Convolutional Neural Network (CNN)
- **Definition**: A type of neural network commonly used for image recognition and processing tasks. It uses convolutional layers to capture spatial hierarchies in data.

### Dense Layer
- **Definition**: A fully connected neural network layer where each neuron is connected to every neuron in the previous layer.

### Epoch
- **Definition**: A complete pass through the entire training dataset by the learning algorithm.

### Loss Function
- **Definition**: A function that measures how far off a model's predictions are from the actual values. The goal is to minimize this function during training.

### ReLU (Rectified Linear Unit)
- **Definition**: A common activation function that outputs the input if it's positive, and zero otherwise.

### Sigmoid
- **Definition**: A common activation function used for binary classification tasks that maps the input to a value between 0 and 1.

### Softmax
- **Definition**: An activation function often used in the output layer of classification networks. It converts the raw outputs into probabilities for each class.

### Neural Network
- **Definition**: A computational model composed of layers of interconnected neurons that are used to approximate complex functions.

### Weight
- **Definition**: The adjustable parameters in a neural network that are updated during training to minimize the loss function.
