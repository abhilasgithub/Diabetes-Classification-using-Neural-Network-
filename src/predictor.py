import numpy as np

from tensorflow import keras


class Predictor:

    def __init__(self, model_path, scaler):

        self.model = keras.models.load_model(model_path)

        self.scaler = scaler

    def predict(self, patient_data):

        patient_data = np.array([patient_data])

        patient_data = self.scaler.transform(
            patient_data
        )

        prediction = self.model.predict(
            patient_data
        )

        probability = prediction[0][0]

        return probability

    def get_result(self, probability):

        if probability >= 0.5:
            return "Positive"

        return "Negative"