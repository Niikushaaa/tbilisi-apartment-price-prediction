import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Dataset-ის ჩატვირთვა
df = pd.read_csv("data/dataset.csv")

# Input features და target
X = df[["area", "rooms", "district", "floor", "condition", "parking"]]
y = df["price"]

# რიცხვითი და კატეგორიული სვეტები
numeric_features = ["area", "rooms", "floor"]
categorical_features = ["district", "condition", "parking"]

# კატეგორიული მონაცემების გარდაქმნა OneHotEncoder-ით
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

# მოდელის pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# Train/Test split
test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

# MLflow Experiment
mlflow.set_experiment("Tbilisi Apartment Price Prediction")

with mlflow.start_run() as run:
    # მოდელის სწავლება
    model.fit(X_train, y_train)

    # პროგნოზი
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Parameters
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # Metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Model Logging
    mlflow.sklearn.log_model(model, "model")

    print("Model trained and logged successfully!")
    print("Run ID:", run.info.run_id)
    print("MAE:", mae)
    print("R2 Score:", r2)
    print("Run ID:", run.info.run_id)