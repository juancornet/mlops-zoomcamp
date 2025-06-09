#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import root_mean_squared_error

import pickle
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

import pandas as pd
import sys
import os

def train(year = 2023, month = 3):
    df = pd.read_parquet(f'data/preprocessed/nyc_taxi_{year}_{month:02d}_preprocessed.parquet')

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    print(f"Training model on {len(df)} rows")

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    rmse = root_mean_squared_error(y_train, y_pred)

    print(f"RMSE on train: {rmse}")
    print(f"Intercept: {lr.intercept_}")

    # Save the model or results if needed
    os.makedirs('models', exist_ok=True)
    
    
    with open("models/preprocessor.bin", "wb") as f_out:
        pickle.dump(dv, f_out)
    
    model_path = f'models/model_{year}_{month:02d}.bin'

    with open(model_path, 'wb') as f:
        pickle.dump((lr), f)    

    with mlflow.start_run():

        mlflow.log_param("train-data-path", 'data/preprocessed/nyc_taxi_{year}_{month:02d}_preprocessed.parquet')

        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact("models/preprocessor.bin", artifact_path="preprocessor")
        mlflow.log_artifact(local_path=model_path, artifact_path="model")

        mlflow.sklearn.log_model(lr, artifact_path="model", registered_model_name="nyc-taxi-model")

    return 

if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    train(year, month)