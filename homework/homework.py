import gzip
import json
import os
import pickle
import zipfile
from glob import glob

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore


def load_data(input_directory):
    """Read zip files in a directory and store them in dfs"""
    dfs = []

    routes = glob(f"{input_directory}/*")

    for route in routes:
        with zipfile.ZipFile(f"{route}", mode="r") as zf:
            for fn in zf.namelist():

                with zf.open(fn) as f:
                    dfs.append(pd.read_csv(f, sep=",", index_col=0))

    return dfs


def _create_output_directory(output_directory):
    """Create output directory if it does not exist."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


def clean_df(df):
    """Paso 1: Clean dataset"""
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    return df.dropna()


def split_data(df):
    """Paso 2: feats & Target"""
    return df.drop(columns=["default"]), df["default"]


def create_pipeline():
    """Paso 3: Create processing pipeline"""
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )


def create_estimator(pipeline):
    """Pipeline"""
    param_grid = {
        "classifier__n_estimators": [100, 200, 500],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )


def _save_model(path, estimator):
    _create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def calculate_metrics(dataset_type, y_true, y_pred):
    """Calculate metrics"""
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion(dataset_type, y_true, y_pred):
    """Confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def _run_jobs():
    # Crear directorios de salida
    _create_output_directory("files/output")
    _create_output_directory("files/models/")

    test_df, train_df = [clean_df(df) for df in load_data("files/input")]

    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    pipeline = create_pipeline()

    estimator = create_estimator(pipeline)
    estimator.fit(x_train, y_train)

    _save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calculate_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calculate_metrics("train", y_train, y_train_pred)

    test_confusion_metrics = calculate_confusion("test", y_test, y_test_pred)
    train_confusion_metrics = calculate_confusion("train", y_train, y_train_pred)

    # Guardar las métricas en el archivo
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")


if __name__ == "__main__":
    _run_jobs()
