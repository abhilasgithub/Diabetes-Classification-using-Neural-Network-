from data_processor import DataProcessor
from model_trainer import ModelTrainer
from predictor import Predictor

class DiabetesPredictionApp:

    def __init__(self):

        self.data_processor = DataProcessor(
            "../data/diabetes.csv"
        )

    def train(self):

        # Load dataset
        data = self.data_processor.load_data()

        # Preprocess dataset
        X_train, X_test, y_train, y_test = (
            self.data_processor.preprocess_data(data)
        )

        # Create model
        trainer = ModelTrainer(
            X_train.shape[1]
        )

        trainer.build_model()

        # Train model
        trainer.train_model(
            X_train,
            y_train
        )

        # Evaluate model
        trainer.evaluate_model(
            X_test,
            y_test
        )

        # Save model
        trainer.save_model(
            "../models/diabetes_model.keras"
        )

    def predict(self):

        # Load dataset
        data = self.data_processor.load_data()

        # Create scaler
        self.data_processor.preprocess_data(data)

        predictor = Predictor(
            "../models/diabetes_model.keras",
            self.data_processor.scaler
        )

        # Example patient
        patient = [
            6,       # Pregnancies
            148,     # Glucose
            72,      # BloodPressure
            35,      # SkinThickness
            0,       # Insulin
            33.6,    # BMI
            0.627,   # DiabetesPedigreeFunction
            50       # Age
        ]

        probability = predictor.predict(
            patient
        )

        result = predictor.get_result(
            probability
        )

        print("\nProbability:", probability)
        print("Prediction:", result)


if __name__ == "__main__":

    app = DiabetesPredictionApp()

    app.train()

    app.predict()