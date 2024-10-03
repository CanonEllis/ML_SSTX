
# Assignment: Implementing Multilayer Perceptrons (MLP) for the MNIST Dataset

## Objective:
Students will progressively build Multilayer Perceptrons (MLPs) to classify images from the MNIST dataset, starting with a 1-layer MLP and extending up to a 4-layer MLP. The assignment will involve implementing forward propagation, backpropagation, and training the MLPs.

---

## Task 1: Load and Preprocess the MNIST Dataset

1. **Download and Preprocess the MNIST Dataset**:
   - Use a library like `tensorflow` or `keras` to download and preprocess the MNIST dataset.
   - Normalize the pixel values to be in the range \([0, 1]\).
   
   Example code to load and normalize the MNIST dataset:
   ```python
   import tensorflow as tf
   from tensorflow.keras.datasets import mnist

   # Load the dataset
   (x_train, y_train), (x_test, y_test) = mnist.load_data()

   # Normalize the dataset
   x_train = x_train.astype('float32') / 255.0
   x_test = x_test.astype('float32') / 255.0

   # Flatten the images (28x28 -> 784)
   x_train = x_train.reshape(-1, 28*28)
   x_test = x_test.reshape(-1, 28*28)

   # Convert labels to one-hot vectors
   y_train = tf.keras.utils.to_categorical(y_train, 10)
   y_test = tf.keras.utils.to_categorical(y_test, 10)
   ```

---

## Task 2: Implement a 1-Layer MLP

- **Description**: Implement a simple neural network with just one layer (input and output). The network should be fully connected and use softmax activation at the output layer.
  
- **Architecture**: 
  - Input: 784 neurons (28x28 pixel images).
  - Output: 10 neurons (one for each class).
  
- **Example Code**:
   ```python
   import numpy as np

   def softmax(z):
       exp_z = np.exp(z - np.max(z))
       return exp_z / exp_z.sum(axis=1, keepdims=True)

   def initialize_params(input_size, output_size):
       np.random.seed(42)
       W = np.random.randn(input_size, output_size) * 0.01
       b = np.zeros((1, output_size))
       return W, b

   def forward_propagation(X, W, b):
       Z = np.dot(X, W) + b
       A = softmax(Z)
       return A

   # Parameters
   W, b = initialize_params(784, 10)
   A = forward_propagation(x_train[:100], W, b)
   ```

---

## Task 3: Implement a 2-Layer MLP

- **Description**: Now add a hidden layer with a non-linear activation function (ReLU). This will introduce the capability to learn more complex patterns.
  
- **Architecture**: 
  - Input: 784 neurons (28x28 pixel images).
  - Hidden Layer: 128 neurons (with ReLU activation).
  - Output Layer: 10 neurons (with softmax activation).

- **Example Code**:
   ```python
   def relu(z):
       return np.maximum(0, z)

   def initialize_2layer_params(input_size, hidden_size, output_size):
       np.random.seed(42)
       W1 = np.random.randn(input_size, hidden_size) * 0.01
       b1 = np.zeros((1, hidden_size))
       W2 = np.random.randn(hidden_size, output_size) * 0.01
       b2 = np.zeros((1, output_size))
       return W1, b1, W2, b2

   def forward_propagation_2layer(X, W1, b1, W2, b2):
       Z1 = np.dot(X, W1) + b1
       A1 = relu(Z1)
       Z2 = np.dot(A1, W2) + b2
       A2 = softmax(Z2)
       return A1, A2

   # Initialize parameters
   W1, b1, W2, b2 = initialize_2layer_params(784, 128, 10)
   A1, A2 = forward_propagation_2layer(x_train[:100], W1, b1, W2, b2)
   ```

---

## Task 4: Pseudocode and Math for 3-Layer and 4-Layer MLPs

### 3-Layer MLP (with 2 hidden layers)

- **Architecture**:
  - Input: 784 neurons
  - First Hidden Layer: 128 neurons (ReLU)
  - Second Hidden Layer: 64 neurons (ReLU)
  - Output Layer: 10 neurons (Softmax)

- **Pseudocode**:
   1. Initialize weights and biases for each layer.
   2. Forward Propagation:
      - `Z1 = X @ W1 + b1` 
      - `A1 = ReLU(Z1)`
      - `Z2 = A1 @ W2 + b2`
      - `A2 = ReLU(Z2)`
      - `Z3 = A2 @ W3 + b3`
      - `A3 = Softmax(Z3)`
   3. Backpropagation:
      - Compute gradients of the loss with respect to each layer's weights and biases.
   4. Update weights using gradient descent.

### 4-Layer MLP (with 3 hidden layers)

- **Architecture**:
  - Input: 784 neurons
  - First Hidden Layer: 256 neurons (ReLU)
  - Second Hidden Layer: 128 neurons (ReLU)
  - Third Hidden Layer: 64 neurons (ReLU)
  - Output Layer: 10 neurons (Softmax)

- **Pseudocode**:
   1. Initialize weights and biases for each layer.
   2. Forward Propagation:
      - `Z1 = X @ W1 + b1` 
      - `A1 = ReLU(Z1)`
      - `Z2 = A1 @ W2 + b2`
      - `A2 = ReLU(Z2)`
      - `Z3 = A2 @ W3 + b3`
      - `A3 = ReLU(Z3)`
      - `Z4 = A3 @ W4 + b4`
      - `A4 = Softmax(Z4)`
   3. Backpropagation:
      - Compute gradients of the loss with respect to each layer's weights and biases.
   4. Update weights using gradient descent.

---

## Challenge Question:

### Adding Dropout Regularization

- **Description**: Modify your 4-layer MLP by adding **Dropout** between each layer to prevent overfitting. Use a dropout rate of 0.2 for the hidden layers.
  
- **Instructions**:
   1. Implement dropout during the forward pass by randomly setting a fraction of the activations in each hidden layer to zero.
   2. Adjust the remaining activations to maintain the expected output.
   3. Train your dropout-regularized MLP on the MNIST dataset and report the accuracy.

---

## Submission:

- Python notebook implementing each MLP (1-layer, 2-layer, 3-layer, and 4-layer) with clear comments.
- A report with results for each MLP architecture, including training accuracy, validation accuracy, and any observations on how the model performance changes with more layers.
- For the challenge question, include a discussion on how dropout regularization impacted your model's performance.
