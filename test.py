from dataset import KMNISTDataset
from model import KMNISTModel

class KMNISTTest:
    def __init__(self, train_images_path, train_labels_path, test_images_path, test_labels_path):
        self.dataset = KMNISTDataset(train_images_path, train_labels_path, test_images_path, test_labels_path)
        self.dataset.preprocess()
        self.model = KMNISTModel()

    def train_model(self, epochs=5, batch_size=32):
        self.model.model.fit(self.dataset.train_images, self.dataset.train_labels, epochs=epochs, batch_size=batch_size, validation_data=(self.dataset.test_images, self.dataset.test_labels))

    def evaluate_model(self):
        test_loss, test_acc = self.model.model.evaluate(self.dataset.test_images, self.dataset.test_labels)
        print(f'Test accuracy: {test_acc}')

    def save_model(self, model_save_path):
        self.model.save_model(model_save_path)

    def predict_images(self, input_data):
        predictions = self.model.model.predict(input_data)
        return predictions


# Define paths to dataset files
train_images_path = './assets/k49-train-imgs.npz'
train_labels_path = './assets/k49-train-labels.npz'
test_images_path = './assets/k49-test-imgs.npz'
test_labels_path = './assets/k49-test-labels.npz'

# Create an instance of KMNISTApp
app = KMNISTTest(train_images_path, train_labels_path, test_images_path, test_labels_path)

# Train the model
app.train_model()

# Evaluate the model
app.evaluate_model()

# Save the model
model_save_path = 'hiragana.model.keras'
app.save_model(model_save_path)

# Perform predictions
# predictions = app.predict_images(input_data)
