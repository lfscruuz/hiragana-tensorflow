import cv2
import numpy as np
from keras import models
from matplotlib import pyplot as plt

class KMNISTRun:
    def __init__(self):
        self.model_path = 'hiragana.model.keras'
        self.class_labels = ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ',
                'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と',
                'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ',
                'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り',
                'る', 'れ', 'ろ', 'わ', 'を', 'う', 'い', 'ゞ', 'あ']
        
        self.model = models.load_model(self.model_path)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(grayscale_image, (28, 28))
        reshaped_image = resized_image[..., np.newaxis]

        normalized_image = reshaped_image / 255.0
        return normalized_image


    def predict_image(self, image_path):
        preprocessed_image = self.preprocess_image(image_path)
        
        input_data = np.expand_dims(preprocessed_image, axis=0)
        input_data = input_data / 255.0
        
        prediction = self.model.predict(input_data)
        return prediction

    def interpret_prediction(self, prediction):

        predicted_class_index = np.argmax(prediction)

        predicted_character = self.class_labels[predicted_class_index]
        return predicted_character
    
    def show_prediction(self):       
        image_paths = ['./images/1.jpg', './images/2.jpg', './images/3.jpg', './images/4.jpg', './images/5.jpg', './images/6.jpg', './images/7.jpg']

        for image_path in image_paths:
            image = cv2.imread(image_path)[:,:,0]
            
            image = np.invert(np.array([image]))
            
            plt.imshow(image[0], cmap=plt.cm.binary)
            plt.show()
            
            prediction = self.predict_image(image_path)
            
            predicted_character = model.interpret_prediction(prediction)
            print(f'O Hiragana digitado foi: {predicted_character}')

model = KMNISTRun()
model.show_prediction()