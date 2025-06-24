import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from train_model import load_and_prepare_data

def load_model(model_path='mnist_model.h5'):
    """Load a trained model."""
    return tf.keras.models.load_model(model_path)

def predict_digit(model, image):
    """Predict the digit in the given image."""
    # Add batch dimension and predict
    prediction = model.predict(np.array([image]))
    # Get the predicted class (digit with highest probability)
    predicted_digit = np.argmax(prediction[0])
    confidence = np.max(tf.nn.softmax(prediction[0]))
    return predicted_digit, confidence

def display_prediction(image, true_label, predicted_digit, confidence):
    """Display the image and prediction results."""
    plt.figure(figsize=(6, 3))
    plt.imshow(image.squeeze(), cmap=plt.cm.binary)
    plt.title(f'True: {true_label}, Predicted: {predicted_digit} ({confidence:.2%})')
    plt.colorbar()
    plt.show()

def main():
    # Load the trained model
    try:
        model = load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure to train the model first by running 'python train_model.py'")
        return
    
    # Load test data
    (_, _), (test_images, test_labels) = load_and_prepare_data()
    
    # Make predictions on a few test images
    num_predictions = 5
    print(f"\nMaking predictions on {num_predictions} random test images...")
    
    for i in range(num_predictions):
        # Select a random test image
        idx = np.random.randint(0, len(test_images))
        test_image = test_images[idx]
        true_label = test_labels[idx]
        
        # Make prediction
        predicted_digit, confidence = predict_digit(model, test_image)
        
        # Display results
        print(f"\nTest image {i+1}:")
        print(f"- True digit: {true_label}")
        print(f"- Predicted digit: {predicted_digit}")
        print(f"- Confidence: {confidence:.2%}")
        
        # Display the image and prediction
        display_prediction(test_image, true_label, predicted_digit, confidence)

if __name__ == "__main__":
    main()
