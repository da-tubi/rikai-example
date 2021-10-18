import getpass

import mlflow
import rikai
from pyspark.ml.regression import IsotonicRegression

from example import spark

mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# enable autologging
# don't know how to enable it for spark yet
# mlflow.spark.autolog()

# originally from https://spark.apache.org/docs/latest/ml-classification-regression.html#isotonic-regression

# Loads data.
dataset = spark.read.format("libsvm") \
    .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")

with mlflow.start_run() as run:
    ####
    # Part 1: Train the model and register it on MLflow
    ####

    model = IsotonicRegression().fit(dataset)
    print("Boundaries in increasing order: %s\n" % str(model.boundaries))
    print("Predictions associated with the boundaries: %s\n" % str(model.predictions))

    schema = "float"
    registered_model_name = f"{getpass.getuser()}_spark_ir"

    rikai.mlflow.spark.log_model(
        model,
        "model",
        schema,
        registered_model_name=registered_model_name)

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
    df = dataset.toDF("label", "features")
    df.createOrReplaceTempView("tbl_X")

    # TODO
    # pyspark.sql.utils.AnalysisException: cannot resolve 'array(tbl_x.`label`, tbl_x.`features`)' due to data type mismatch:
    # input to function array should all be the same type,
    # but it's [double, struct<type:tinyint,size:int,indices:array<int>,values:array<double>>]; line 2 pos 40;
    result = spark.sql(f"""
    select ML_PREDICT(mlflow_sklearn_m, array(label, features)) from tbl_X
    """)

    result.printSchema()
    result.show(10, vertical=False, truncate=False)

    # df.show()
    # Makes predictions.
    # model.transform(dataset).show()
