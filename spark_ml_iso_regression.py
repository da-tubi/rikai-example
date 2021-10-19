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

    df.show()

    # : org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 8.0 failed 1 times, most recent failure: Lost task 0.0 in stage 8.0 (TID 8) (192.168.1.241 executor driver): java.lang.UnsupportedOperationException: Unsupported data type: struct<type:tinyint,size:int,indices:array<int>,values:array<double>>
    # 	at org.apache.spark.sql.util.ArrowUtils$.toArrowType(ArrowUtils.scala:57)

    #   def toArrowType(dt: DataType, timeZoneId: String): ArrowType = dt match {
    #     case BooleanType => ArrowType.Bool.INSTANCE
    #     case ByteType => new ArrowType.Int(8, true)
    #     case ShortType => new ArrowType.Int(8 * 2, true)
    #     case IntegerType => new ArrowType.Int(8 * 4, true)
    #     case LongType => new ArrowType.Int(8 * 8, true)
    #     case FloatType => new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    #     case DoubleType => new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    #     case StringType => ArrowType.Utf8.INSTANCE
    #     case BinaryType => ArrowType.Binary.INSTANCE
    #     case DecimalType.Fixed(precision, scale) => new ArrowType.Decimal(precision, scale)
    #     case DateType => new ArrowType.Date(DateUnit.DAY)
    #     case TimestampType =>
    #       if (timeZoneId == null) {
    #         throw new UnsupportedOperationException(
    #           s"${TimestampType.catalogString} must supply timeZoneId parameter")
    #       } else {
    #         new ArrowType.Timestamp(TimeUnit.MICROSECOND, timeZoneId)
    #       }
    #     case _ =>
    #       throw new UnsupportedOperationException(s"Unsupported data type: ${dt.catalogString}")
    #   }
    result = spark.sql(f"""
    select ML_PREDICT(mlflow_sklearn_m, named_struct('label',label,'features' ,features)) from tbl_X
    """)

    result.printSchema()
    result.show(10, vertical=False, truncate=False)

    # df.show()
    # Makes predictions.
    # model.transform(dataset).show()
