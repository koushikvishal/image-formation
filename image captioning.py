import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def prepare_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to 224x224 for ResNet50
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess for ResNet50
    return img_array

def classify_image(img_path):
    """Classify the image and display the result."""
    img_array = prepare_image(img_path)
    
    # Predict the class probabilities for the image
    predictions = model.predict(img_array)
    
    # Decode predictions to human-readable labels
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Top 3 predictions
    
    print(f"Predictions for image {img_path}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i+1}: {label} ({score*100:.2f}%)")
    
    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Prediction: {decoded_predictions[0][1]}")
    plt.axis('off')
    plt.show()

# Example usage
img_path = 'dog.jpg'  # Replace this with the path to your image
classify_image(img_path)
