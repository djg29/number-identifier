# Handwritten Digit Recognition

A simple neural network model for recognizing handwritten digits using the MNIST dataset, built with TensorFlow and Keras.

## Features

- Train a Convolutional Neural Network (CNN) to recognize handwritten digits
- Evaluate model performance on test data
- Make predictions on new handwritten digits
- Visualize training progress and predictions

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model, run:

```bash
python train_model.py
```

This will:
1. Download and preprocess the MNIST dataset
2. Create and train a CNN model
3. Save the trained model as `mnist_model.h5`
4. Display training history and test accuracy

### Making Predictions

To make predictions using the trained model, run:

```bash
python predict.py
```

This will:
1. Load the trained model
2. Randomly select test images
3. Display the images along with the model's predictions

## Model Architecture

The model uses the following architecture:

1. Two convolutional layers with max pooling
2. A flattening layer
3. Two dense layers with dropout for regularization
4. Output layer with 10 units (one for each digit 0-9)

## Performance

The model typically achieves:
- Training accuracy: ~99%
- Test accuracy: ~99%

## Files

- `train_model.py`: Script to train the digit recognition model
- `predict.py`: Script to make predictions using the trained model
- `requirements.txt`: List of required Python packages
- `mnist_model.h5`: Trained model (created after training)

## License

This project is open source and available under the MIT License.
