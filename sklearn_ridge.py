import getpass

import mlflow
import numpy as np
import rikai
from sklearn.linear_model import Ridge

from example import spark

mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# enable autologging
mlflow.sklearn.autolog()

# prepare training data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

# prepare evaluation data
X_eval = np.array([[3, 3], [3, 4]])
y_eval = np.dot(X_eval, np.array([1, 2])) + 3

# train a model
model = Ridge(alpha=0.5)
with mlflow.start_run() as run:
    ####
    # Part 1: Train the model and register it on MLflow
    ####
    model.fit(X, y)
    metrics = mlflow.sklearn.eval_and_log_metrics(model, X_eval, y_eval, prefix="val_")

    schema = "float"
    registered_model_name = f"{getpass.getuser()}_sklearn_ridge"
    rikai.mlflow.sklearn.log_model(
        model, "model", schema, registered_model_name=registered_model_name
    )

    ####
    # Part 2: create the model using the registered MLflow uri
    ####
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri)
    spark.sql(
        f"""
    CREATE MODEL mlflow_sklearn_m USING 'mlflow:///{registered_model_name}';
    """
    )

    ####
    # Part 3: predict using the registered Rikai model
    ####
    spark.sql("show models").show(1, vertical=False, truncate=False)

    result = spark.sql(
        f"""
    select ML_PREDICT(mlflow_sklearn_m, array(3, 5))
    """
    )

    result.printSchema()
    result.show(1, vertical=False, truncate=False)
