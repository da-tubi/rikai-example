from sklearn.neighbors import NearestCentroid
import mlflow
import numpy as np
import getpass
import rikai
from example import spark

mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# enable autologging
mlflow.sklearn.autolog()

# prepare training data
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

# train a model
model = NearestCentroid()
with mlflow.start_run() as run:
    ####
    # Part 1: Train the model and register it on MLflow
    ####
    model.fit(X, y)

    schema = "array<int>"
    registered_model_name = f"{getpass.getuser()}_sklearn"
    rikai.mlflow.sklearn.log_model(
        model,
        "model",
        schema,
        registered_model_name = registered_model_name)


    ####
    # Part 2: create the model using the registered MLflow uri
    ####
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri)
    spark.sql(f"""
    CREATE MODEL mlflow_sklearn_m USING 'mlflow:///{registered_model_name}';
    """)

    ####
    # Part 3: predict using the registered Rikai model
    ####
    spark.sql("show models").show(1, vertical=False, truncate=False)

    result = spark.sql(f"""
    select ML_PREDICT(mlflow_sklearn_m, array(array(0.8, -1)))
    """)

    result.printSchema()
    result.show(1, vertical=False, truncate=False)
