# RNN Text Generation with Keras

This project uses a Recurrent Neural Network (RNN) built with Keras to generate short stories from text data. The model is trained on a text file (for example, a book from Project Gutenberg) and learns to predict the next character in a sequence. Once trained, the model can generate new text that resembles the style and structure of the input data.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
   - [Loading and Cleaning Text](#loading-and-cleaning-text)
   - [Creating Sequences](#creating-sequences)
4. [Building the Model](#building-the-model)
5. [Training the Model](#training-the-model)
6. [Generating Text](#generating-text)
7. [Sampling Function](#sampling-function)
8. [Further Resources](#further-resources)

## Overview

This project implements a character-level RNN to generate text. The network uses LSTM layers to capture long-term dependencies in the text. The workflow consists of:
- Reading and preprocessing the text data.
- Dividing the text into overlapping sequences.
- Building and training an LSTM-based model.
- Generating new text using a sampling function that controls randomness.

## Requirements

- Python 3.x
- TensorFlow (with Keras integrated)
- NumPy

You can install the necessary packages using:

```bash
pip install tensorflow numpy
```

## Data Preparation

### Loading and Cleaning Text

The first step is to load your text file (e.g., a Project Gutenberg book) and convert it to lowercase to simplify the character set.

```python
# Load the text from a file
with open("book.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

print("Length of text:", len(text))
```

*Explanation:*
This code reads the entire contents of `book.txt` into a string and converts all characters to lowercase to reduce the vocabulary size.

### Creating Sequences

Next, we create fixed-length sequences of characters. Each sequence will be used to predict the character that follows it.

```python
seq_length = 100  # Length of each sequence

# Create a sorted list of unique characters in the text
chars = sorted(list(set(text)))
print("Total unique characters:", len(chars))

# Create mappings from characters to integers and vice versa
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

# Prepare the sequences and corresponding next characters
sequences = []
next_chars = []
for i in range(0, len(text) - seq_length):
    sequences.append(text[i:i + seq_length])
    next_chars.append(text[i + seq_length])

print("Total sequences:", len(sequences))
```

*Explanation:*
- **Vocabulary Building:** We extract all unique characters and sort them.
- **Mapping:** Two dictionaries are created to map characters to integers and vice versa.
- **Sequence Generation:** We slide a window of `seq_length` over the text to create input sequences and record the next character as the target.

Next, we one-hot encode the input sequences:

```python
import numpy as np

num_sequences = len(sequences)
num_chars = len(chars)

# Initialize the input and output arrays (using boolean arrays for one-hot encoding)
X = np.zeros((num_sequences, seq_length, num_chars), dtype=np.bool_)
y = np.zeros((num_sequences, num_chars), dtype=np.bool_)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1
```

*Explanation:*
The input `X` is a 3D array where each sequence is a matrix of one-hot encoded characters. The target `y` is also one-hot encoded, representing the next character for each sequence.

## Building the Model

We use an LSTM-based model with dropout layers to avoid overfitting. The model predicts the next character given an input sequence.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, num_chars)))
model.add(Dropout(0.2))
model.add(Dense(num_chars, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
```

*Explanation:*
- **LSTM Layer:** Processes the sequential data and captures temporal dependencies.
- **Dropout Layer:** Helps prevent overfitting by randomly setting a fraction of input units to 0.
- **Dense Layer:** Outputs a probability distribution over the next character using a softmax activation.
- **Compilation:** Uses categorical crossentropy as the loss function and the Adam optimizer.

## Training the Model

Train the model with your prepared input and target data. Adjust the number of epochs and batch size based on your dataset and computational resources.

```python
epochs = 20    # Adjust the number of epochs as needed
batch_size = 128

model.fit(X, y, epochs=epochs, batch_size=batch_size)
```

*Explanation:*
The model is trained to minimize the categorical crossentropy loss, which measures the difference between the predicted probability distribution and the actual one-hot encoded target.

## Generating Text

After training, you can generate new text by providing a seed sequence and iteratively predicting the next character.

```python
import random

# Select a random starting index and extract a seed sequence
start_index = random.randint(0, len(text) - seq_length - 1)
generated_text = text[start_index:start_index + seq_length]
print("Seed:", generated_text)

# Generate 400 characters of new text
for i in range(400):
    # Prepare the input by one-hot encoding the current seed sequence
    x_pred = np.zeros((1, seq_length, num_chars))
    for t, char in enumerate(generated_text[-seq_length:]):
        x_pred[0, t, char_to_int[char]] = 1

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, temperature=0.5)  # Temperature controls randomness
    next_char = int_to_char[next_index]

    generated_text += next_char

print("Generated Text:")
print(generated_text)
```

*Explanation:*
- **Seed Selection:** A random segment of the text is chosen as the starting sequence.
- **Prediction Loop:** The model predicts the next character, which is then appended to the generated text.
- **Temperature Parameter:** Adjusts the randomness of predictions.

## Sampling Function

The sampling function converts the model's output probabilities into a specific character index. The `temperature` parameter adjusts the confidence of the predictions.

```python
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature  # Avoid log(0) by adding epsilon
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

*Explanation:*
- **Log Transformation:** The logarithm is applied to the predictions to smooth the distribution.
- **Temperature Scaling:** Dividing by the temperature parameter controls randomness.
- **Exponentiation and Normalization:** Converts log probabilities back into a probability distribution.
- **Sampling:** A character index is sampled from the probability distribution.


# Assignment: RNN Text Generation with Keras

## Assignment Description

In this assignment, you will implement a character-level Recurrent Neural Network (RNN) to generate text. You will select a text dataset from [Project Gutenberg](https://www.gutenberg.org/), preprocess it, and experiment with different RNN architectures to improve text generation quality. Your final submission should include a Jupyter Notebook documenting your process and the best-generated text output from your model.


### Experimentation
Throughout this assignment, you will experiment with different RNN architectures such as:
- Basic RNN
- LSTM
- GRU
- Stacked or bidirectional layers

## Assignment
### Step 1: Dataset Selection
Choose a text dataset from [Project Gutenberg](https://www.gutenberg.org/). It should be a novel, collection of works, genre, or long-form text that provides enough data for training. Download and load the dataset into your Jupyter Notebook.

### Step 2: Data Preprocessing
- Convert all text to lowercase.
- Create a mapping of characters to integers and vice versa.
- Prepare sequences of fixed length (e.g., 100 characters per sequence).
- One-hot encode the sequences for model input.

### Step 3: Model Implementation
- Implement an LSTM-based RNN model as a baseline.
- Train the model on your dataset.
- Generate sample text using your trained model.

### Step 4: Experimentation
Modify the architecture and observe its effects on text generation. Some ideas:
- Increase or decrease the number of LSTM/GRU layers.
- Try a bidirectional LSTM/GRU.
- Adjust dropout rates.
- Tune hyperparameters like learning rate and sequence length.
- Compare LSTM vs. GRU performance.

### Step 5: Submission
Submit a Jupyter Notebook (`.ipynb`) that includes:
1. Your selected dataset and preprocessing steps.
2. A summary of different architectures you experimented with.
3. The best-generated text sample from your model.
4. A short reflection on what worked best and why.

## Challenge Question
Analyze the vanishing gradient problem in the context of training your RNN model. Compare the gradient behavior of a simple RNN, an LSTM, and a GRU by examining gradient magnitudes during training. Which architecture mitigates vanishing gradients most effectively, and why?


## Further Resources

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Project Gutenberg](https://www.gutenberg.org/)
