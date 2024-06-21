import cv2
import numpy as np
from keras import models
from matplotlib import pyplot as plt

class CharacterRecognitionModel:
    def __init__(self, model_path):
        self.model = models.load_model(model_path)

    def preprocess_image(self, image_path):
        # Load image using OpenCV
        image = cv2.imread(image_path)
        # Preprocess image (resize, normalize, etc.)
        # This preprocessing should match the preprocessing done during training
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Resize image to match the input size expected by the model
        resized_image = cv2.resize(grayscale_image, (28, 28))
        # Add channel dimension to the image
        reshaped_image = resized_image[..., np.newaxis]
        # Normalize pixel values
        normalized_image = reshaped_image / 255.0
        return normalized_image


    def predict_image(self, image_path):
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image_path)
        # Reshape and normalize the image
        input_data = np.expand_dims(preprocessed_image, axis=0)
        input_data = input_data / 255.0  # Normalize pixel values if necessary
        # Make prediction
        prediction = self.model.predict(input_data)
        # Process prediction...
        return prediction

    def predict_character(self, prediction):

        # Get the index of the predicted class
        predicted_class_index = np.argmax(prediction)
        # Assuming you have a list of class labels in the same order as your model output
        
        class_labels = ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
                'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と',
                'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ',
                'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り',
                'る', 'れ', 'ろ', 'わ', 'を', 'ん', 'ゝ', 'ゞ', 'ー']
        # Retrieve the predicted character label
        predicted_character = class_labels[predicted_class_index]
        return predicted_character
    
# Usage:
model_path = 'hiragana.model.keras'
image_paths = ['./images/1.jpg', './images/2.jpg', './images/3.jpg', './images/4.jpg', './images/5.jpg', './images/6.jpg', './images/7.jpg']

# Create an instance of the model
model = CharacterRecognitionModel(model_path)


# Make predictions on new images
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)[:,:,0]
    
    # Display the image
    image = np.invert(np.array([image]))
    
    plt.imshow(image[0], cmap=plt.cm.binary)
    plt.show()
    
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Make prediction
    prediction = model.predict_image(image_path)
    
    # Process prediction
    predicted_character = model.predict_character(prediction)
    print(f'O Hiragana digitado foi: {predicted_character}')
