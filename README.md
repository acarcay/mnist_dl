MNIST Digit Classification Model with PyTorch
This project is a Python script that demonstrates how to build, train, and test a simple Artificial Neural Network (ANN) using the popular MNIST handwritten digit dataset. The model is developed using the PyTorch library, and the training process is visualized with matplotlib.

📝 Description
The goal of this project is to present all the steps of a basic deep learning project in a single file, making it ideal for beginners in machine learning. The script performs the following operations in sequence:

Downloads the MNIST dataset and creates DataLoaders.

Visualizes a few sample images from the dataset.

Defines a simple fully-connected neural network model.

Trains the model using the CrossEntropyLoss loss function and the Adam optimizer.

Plots the training loss over the epochs.

Measures the performance of the trained model on the test dataset and reports the accuracy.

✨ Features
Data Handling: Automatically downloads and normalizes the MNIST dataset using torchvision.

Model Architecture: A simple fully-connected neural network with a 784-node input layer, two hidden layers (128 and 64 neurons), and a 10-node output layer (for digits 0-9).

Training Loop: Updates the model's weights using a standard PyTorch training loop.

Visualization: Generates a plot for the training loss and displays sample images from the dataset.

Evaluation: Calculates the final accuracy percentage of the model on the test data.

GPU Support: Automatically utilizes a CUDA-enabled GPU for training if one is available on the system.

⚙️ Requirements
To run this project, you need to have the following libraries installed:

Python (3.6+)

PyTorch

Torchvision

Matplotlib

nstallation
Clone the repository to your local machine or simply download the ann.py file.

It is recommended to create a virtual environment to manage project dependencies:

Bash

python -m venv venv
source venv/bin/activate  # For macOS/Linux
# venv\Scripts\activate   # For Windows
Install the required libraries using pip:

Bash

pip install torch torchvision matplotlib
🏃‍♀️ Usage
Once the installation is complete, you can run the project by executing the following command in your terminal:

Bash

python ann.py
When the script runs, it will:

Download the MNIST dataset to a ./data directory (if not already downloaded).

Open a window displaying 5 sample digits from the dataset.

Begin the training process for 10 epochs, printing the loss for each epoch to the terminal.

After training, open a second window showing a plot of the training loss.

Finally, print the model's final accuracy on the test set to the terminal.
