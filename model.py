from keras import models, layers

class KMNISTModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(49, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(cls, saved_model_path):
        loaded_model = cls()
        loaded_model.model = models.load_model(saved_model_path)
        return loaded_model
