MNIST Handwritten Digit Classification with PyTorch
This project involves creating, training, and testing a simple Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset using PyTorch.

Project Features
Data Handling: Automatically downloads and prepares the MNIST dataset for training using torchvision.

Data Visualization: Includes a function to display random samples and their corresponding labels from the training data.

Model Architecture: Uses a simple neural network composed of fully-connected layers.

Training Loop: Trains the model and prints the average loss value for each epoch to the console.

Loss Curve: Plots a graph showing the change in loss value over epochs after training is complete.

Evaluation: Calculates and displays the accuracy of the model on the test data.

Hardware Support: Automatically uses a CUDA-enabled GPU if available; otherwise, it runs on the CPU.

Model Architecture
The neural network used is a simple feed-forward network consisting of the following layers:

Flatten Layer: Converts 28x28 pixel images into a one-dimensional vector with 784 elements.

Fully Connected Layer 1: Maps 784 input features to 128 output features.

ReLU Activation Function

Fully Connected Layer 2: Maps 128 input features to 64 output features.

ReLU Activation Function

Fully Connected Layer 3 (Output Layer): Maps 64 input features to 10 output features (one class for each digit from 0-9).

The Adam optimization algorithm is used for model optimization, and Cross-Entropy Loss is used as the loss function.

Requirements
To run the project, you will need Python 3 and the following libraries:

PyTorch

Torchvision

Matplotlib

You can install these libraries using pip:

Bash

pip install torch torchvision matplotlib
How to Run
Save the code as a Python file, such as mnist_ann.py.

Open your terminal or command prompt.

Navigate to the directory where you saved the file.

Run the script with the following command:

Bash

python mnist_ann.py
When the script runs, it will first download the dataset (if it hasn't been downloaded before), then display a few sample digits. Afterward, the model will be trained for 10 epochs, a training graph will be plotted, and finally, the accuracy on the test data will be printed to the screen.
