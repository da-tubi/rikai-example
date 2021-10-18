import mlflow
from example import spark
from pyspark.ml.regression import IsotonicRegression

mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)

# enable autologging
mlflow.spark.autolog()

# originally from https://spark.apache.org/docs/latest/ml-classification-regression.html#isotonic-regression

# Loads data.
dataset = spark.read.format("libsvm") \
    .load("data/mllib/sample_isotonic_regression_libsvm_data.txt")

# Trains an isotonic regression model.
model = IsotonicRegression().fit(dataset)
print("Boundaries in increasing order: %s\n" % str(model.boundaries))
print("Predictions associated with the boundaries: %s\n" % str(model.predictions))

# Makes predictions.
model.transform(dataset).show()
