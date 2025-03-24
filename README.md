# Handwrite Digits Recognition

This project is a handwritten digits recognition system using a neural network built with TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Introduction
This project uses the MNIST dataset to train a neural network to recognize handwritten digits. The model is built using TensorFlow and Keras, and it achieves high accuracy on the test dataset.

## Installation
To run this project, you need to have Python and TensorFlow installed. You can set up the environment using the provided virtual environment in the `tf` directory.

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd Handwrite_digits_recognition
    ```

2. Activate the virtual environment:
    ```sh
    source tf/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To train and evaluate the model, run the `main.py` script:
```sh
python3 main.py
```

The script will train the model on the MNIST dataset, save the trained model to handwrittenRecognition.keras, and evaluate the model on the test dataset.

## Model Architecture
The neural network model consists of the following layers:

* Flatten layer to convert the 28x28 images into a 1D array
* Two Dense layers with 128 neurons each and ReLU activation
* Output Dense layer with 10 neurons and softmax activation

## Results

After training for 30 epochs with a batch size of 128, the model achieves high accuracy on the test dataset. The loss and accuracy are printed at the end of the training process.

## License
This project is licensed under the MIT License.