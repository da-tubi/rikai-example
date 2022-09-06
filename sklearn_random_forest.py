import getpass

import mlflow
import numpy as np
import rikai
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from example import spark

mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# enable autologging
mlflow.sklearn.autolog()

X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

# train a model
model = RandomForestClassifier(max_depth=2, random_state=0)
with mlflow.start_run() as run:
    ####
    # Part 1: Train the model and register it on MLflow
    ####
    model.fit(X, y)

    schema = "int"
    registered_model_name = f"{getpass.getuser()}_sklearn"
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
    select ML_PREDICT(mlflow_sklearn_m, array(0,0,0,0))
    """
    )

    result.printSchema()
    result.show(1, vertical=False, truncate=False)
