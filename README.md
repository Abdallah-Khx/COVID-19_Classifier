# COVID-19 X-Ray Classification with MLflow

This project focuses on developing a deep learning model that classifies chest X-ray images into four distinct classes: COVID-19, Lung Opacity, Normal, and Viral Pneumonia. To manage experiment tracking, model registry, and model serving, I'm utilizing MLflow.

## Dataset

The project uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data) dataset, which contains chest X-ray scans organized into four categories:

- COVID-19: X-ray images of COVID-19 positive cases
- Lung Opacity: X-ray images showing lung opacity (Non-COVID lung infection)
- Normal: X-ray images of normal cases
- Viral Pneumonia: X-ray images of viral pneumonia cases

Each category contains real X-ray scans and corresponding metadata files in Excel format providing additional information about the cases.

## Project Structure

```
COVID-19-ML-Flow/
├── app/
│   ├── main.py              # FastAPI application
│   └── static/              # Frontend static files
├── python/
│   ├── model.py             # Model architecture definition
│   ├── preprocessing.py     # Data preprocessing utilities
│   └── train.py             # Training script
├── COVID-19_Radiography_Dataset/
│   ├── COVID/
│   ├── Normal/
│   ├── Lung_Opacity/
│   └── Viral Pneumonia/
└── requirements.txt         # Project dependencies
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- MLflow server running locally

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Abdallah-Khx/COVID-19_Classifier.git
cd COVID-19_Classifier
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

## Model Training

1. Train the model:

```bash
python python/train.py
```

2. Register the model in MLflow Model Registry:

```bash
mlflow models register-model -m "runs:/<RUN_ID>/model" -n "covid_classifier"
```

3. Transition the model to Production:

```bash
mlflow model-registry transition-model-stage -m "covid_classifier" -v 1 -s Production
```

## Model Serving

The trained model can be served in two ways:

1. From MLflow Model Registry:

```python
import mlflow
model = mlflow.pytorch.load_model("models:/covid_classifier/Production")
```

2. From local storage:

```python
import torch
model = torch.load("covid_classifier.pth")
```

## Model Architecture

- Based on ResNet architecture
- Modified for 4-class classification
- Transfer learning from pre-trained weights
- Cross-validation training
- Early stopping and learning rate scheduling

## Development

1. Make changes to the model architecture in `python/model.py`
2. Update preprocessing in `python/preprocessing.py`
3. Retrain the model using `python/train.py`
4. Register the new model version in MLflow
5. Test the model using FastAPI application
