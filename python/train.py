import os
import torch
import mlflow
import numpy as np
from datetime import datetime
from preprocessing import DataPreprocessor
from model import COVIDClassifier, ModelTrainer

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("COVID-19_XRay_Classification")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    split_dir = "splits"

    # Uncomment the next line to create splits (run once, then comment again)
    preprocessor.split_and_save(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "COVID-19_Radiography_Dataset")), split_dir=split_dir)

    # Create data loaders from split
    train_loader, val_loader, test_loader = preprocessor.create_data_loaders_from_split(
        split_dir=split_dir, batch_size=32, num_workers=4
    )

    # Initialize model and trainer
    model = COVIDClassifier()
    trainer = ModelTrainer(model, device)

    # Start MLflow run
    with mlflow.start_run(run_name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log data preprocessing parameters
        preprocessor.log_preprocessing_params()

        # Train the model
        model = trainer.train(
            train_loader,
            val_loader,
            epochs=20
        )

        # Log model
        example_input = np.random.randn(1, 3, 299, 299).astype(np.float32)  # Numpy array
        mlflow.pytorch.log_model(model, "model", input_example=example_input)

if __name__ == "__main__":
    main() 