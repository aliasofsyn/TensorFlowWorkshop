# MNIST Handwritten Digit Classification using TensorFlow

## Overview
This project uses a neural network implemented with TensorFlow to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of digits (0-9) and is a popular dataset for testing image classification models.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Installation
To run this project, you need to have Python 3.8+ and pip installed.

### Step 1: Clone the repository
```bash
git clone <https://github.com/aliasofsyn/TensorFlowWorkshop>
cd <repository-folder>
```

### Step 2: Set up a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  
```
### Step 3: Installing Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Dataset
The MNIST dataset contains 70,000 images of handwritten digits, split into:

- Training set: 60,000 images
- Test set: 10,000 images
- Each image is 28x28 pixels, with pixel values ranging from 0 to 255.

### Step 5: Preprocessing
The preprocessing steps applied to the dataset include:

Normalization: The pixel values are scaled from [0, 255] to [0, 1] for faster convergence.
Shuffling and Batching: The training data is shuffled and split into batches of size 128 to improve training efficiency.

### Step 6: Model Architecture
The neural network model consists of:

1. Input Layer: Flattens the 28x28 images into a 1D array of 784 features.
2. Hidden Layer: A dense layer with 128 neurons and ReLU activation.
3. Output Layer: A dense layer with 10 neurons (representing digits 0-9) and no activation function, as we use the from_logits=True setting in the loss function.

### Results
- Training Accuracy: Achieved ~98.9% accuracy after 8 epochs.
- Validation Accuracy: Achieved ~97.7% accuracy after 8 epochs.
The learning curve is visualized using Matplotlib to show training and validation accuracy across 8 epochs.

### References
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview)

