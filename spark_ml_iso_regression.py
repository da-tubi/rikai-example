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
    #  TypeError: cannot pickle 'generator' object
    # 	at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.handlePythonException(PythonRunner.scala:517)
    # 	at org.apache.spark.sql.execution.python.PythonUDFRunner$$anon$2.read(PythonUDFRunner.scala:84)
    # 	at org.apache.spark.sql.execution.python.PythonUDFRunner$$anon$2.read(PythonUDFRunner.scala:67)
    # 	at org.apache.spark.api.python.BasePythonRunner$ReaderIterator.hasNext(PythonRunner.scala:470)
    # 	at org.apache.spark.InterruptibleIterator.hasNext(InterruptibleIterator.scala:37)
    # 	at scala.collection.Iterator$$anon$11.hasNext(Iterator.scala:489)
    # 	at scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:458)
    # 	at scala.collection.Iterator$$anon$10.hasNext(Iterator.scala:458)
    # 	at org.apache.spark.sql.catalyst.expressions.GeneratedClass$GeneratedIteratorForCodegenStage1.processNext(Unknown Source)
    # 	at org.apache.spark.sql.execution.BufferedRowIterator.hasNext(BufferedRowIterator.java:43)
    # 	at org.apache.spark.sql.execution.WholeStageCodegenExec$$anon$1.hasNext(WholeStageCodegenExec.scala:755)
    # 	at org.apache.spark.sql.execution.SparkPlan.$anonfun$getByteArrayRdd$1(SparkPlan.scala:345)
    # 	at org.apache.spark.rdd.RDD.$anonfun$mapPartitionsInternal$2(RDD.scala:898)
    # 	at org.apache.spark.rdd.RDD.$anonfun$mapPartitionsInternal$2$adapted(RDD.scala:898)
    # 	at org.apache.spark.rdd.MapPartitionsRDD.compute(MapPartitionsRDD.scala:52)
    # 	at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:373)
    # 	at org.apache.spark.rdd.RDD.iterator(RDD.scala:337)
    # 	at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:90)
    # 	at org.apache.spark.scheduler.Task.run(Task.scala:131)
    # 	at org.apache.spark.executor.Executor$TaskRunner.$anonfun$run$3(Executor.scala:497)
    # 	at org.apache.spark.util.Utils$.tryWithSafeFinally(Utils.scala:1439)
    # 	at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:500)
    # 	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    # 	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    # 	at java.lang.Thread.run(Thread.java:748)

    result = spark.sql(f"""
    select ML_PREDICT(mlflow_sklearn_m, struct(label, features)) from tbl_X
    """)

    result.printSchema()
    result.show(10, vertical=False, truncate=False)

    # df.show()
    # Makes predictions.
    # model.transform(dataset).show()
