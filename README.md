# Tbilisi Apartment Price Prediction

## Project Description

This project predicts apartment prices in Tbilisi using Machine Learning.

The model uses apartment features such as area, number of rooms, district, floor, condition, and parking availability to predict the apartment price.

This is a regression task because the target variable is a continuous numeric value: apartment price.

## Dataset

The dataset contains apartment information with the following features:

- area
- rooms
- district
- floor
- condition
- parking

Target variable:

- price

## Technologies Used

- Python
- Pandas
- Scikit-learn
- MLflow
- FastAPI
- Uvicorn

## ML Model

The project uses a Linear Regression model as a baseline model.

Categorical features are processed using OneHotEncoder.

The model is trained using a Scikit-learn Pipeline.

## MLflow

MLflow is used for experiment tracking and model logging.

Logged information:

- Experiment
- Parameters
- Metrics
- Trained model

Parameters:

- model_type
- test_size
- random_state

Metrics:

- MAE
- R2 Score

## FastAPI

The project includes a FastAPI service with the following endpoint:

```text
POST /predict