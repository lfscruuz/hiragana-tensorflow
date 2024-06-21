import numpy as np

class KMNISTDataset:
    def __init__(self, train_images_path, train_labels_path, test_images_path, test_labels_path):
        self.train_images = np.load(train_images_path)['arr_0']
        self.train_labels = np.load(train_labels_path)['arr_0']
        self.test_images = np.load(test_images_path)['arr_0']
        self.test_labels = np.load(test_labels_path)['arr_0']

    def preprocess(self):
        # Normalize pixel values to the range [0, 1]
        self.train_images = self.train_images.astype('float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0
