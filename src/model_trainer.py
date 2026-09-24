from tensorflow import keras
from tensorflow.keras import layers


class ModelTrainer:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):

        self.model = keras.Sequential([

            layers.Input(shape=(self.input_shape,)),

            layers.Dense(32, activation="relu"),

            layers.Dense(16, activation="relu"),

            layers.Dense(1, activation="sigmoid")
        ])

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return self.model

    def train_model(self, X_train, y_train):

        history = self.model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.2
        )

        return history

    def evaluate_model(self, X_test, y_test):

        loss, accuracy = self.model.evaluate(
            X_test,
            y_test
        )

        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def save_model(self, path):

        self.model.save(path)

        print("Model saved:", path)