import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Load and prepare the MNIST dataset."""
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Add a channels dimension (needed for the CNN)
    train_images = train_images[..., tf.newaxis].astype("float32")
    test_images = test_images[..., tf.newaxis].astype("float32")
    
    return (train_images, train_labels), (test_images, test_labels)

def create_model():
    """Create a simple CNN model for digit recognition."""
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten the output for the dense layers
        layers.Flatten(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # Dropout for regularization
        layers.Dense(10)  # 10 output classes (digits 0-9)
    ])
    
    return model

def train_model(model, train_images, train_labels, test_images, test_labels, epochs=5):
    """Train the model and return the training history."""
    # Compile the model
    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels)
    )
    
    return history

def evaluate_model(model, test_images, test_labels):
    """Evaluate the model on test data and print the accuracy."""
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")
    return test_acc

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def save_model(model, model_name='mnist_model'):
    """Save the trained model."""
    model.save(f'{model_name}.h5')
    print(f"Model saved as {model_name}.h5")

def main():
    print("Loading and preparing data...")
    (train_images, train_labels), (test_images, test_labels) = load_and_prepare_data()
    
    print("Creating model...")
    model = create_model()
    model.summary()
    
    print("\nTraining model...")
    history = train_model(model, train_images, train_labels, test_images, test_labels, epochs=5)
    
    print("\nEvaluating model...")
    test_accuracy = evaluate_model(model, test_images, test_labels)
    
    # Save the model
    save_model(model)
    
    # Plot training history
    plot_training_history(history)
    
    return model, test_accuracy

if __name__ == "__main__":
    model, test_accuracy = main()
