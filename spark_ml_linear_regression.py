from pyspark.ml.regression import LinearRegression
from example import spark
import mlflow
import getpass
import rikai


mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)

with mlflow.start_run() as run:
    # Load training data
    training = spark.read.format("libsvm") \
        .load("data/mllib/sample_linear_regression_data.txt")

    training.show()
    training.printSchema()

    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(training)
    schema = "float"
    registered_model_name = f"{getpass.getuser()}_spark_ir"
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri)
    spark.sql(f"""
    CREATE MODEL mlflow_spark_m USING 'mlflow:///{registered_model_name}';
    """)

    rikai.mlflow.spark.log_model(
        lrModel,
        "model",
        schema,
        registered_model_name=registered_model_name)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lrModel.coefficients))
    print("Intercept: %s" % str(lrModel.intercept))

    # Summarize the model over the training set and print out some metrics
    trainingSummary = lrModel.summary
    print("numIterations: %d" % trainingSummary.totalIterations)
    print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
    trainingSummary.residuals.show()
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)

    spark.sql("show models").show(1, vertical=False, truncate=False)
    df = training.toDF("label", "features")
    df.createOrReplaceTempView("tbl_X")

    df.show()

    result = spark.sql(f"""
    select ML_PREDICT(mlflow_spark_m, array(1,1,1,1,1,1,1,1,1,1)) from tbl_X
    """)


    result.printSchema()
    # 21/10/19 10:44:40 ERROR TaskSetManager: Task 0 in stage 10.0 failed 1 times; aborting job
    # Traceback (most recent call last):
    #   File "spark_ml_linear_regression.py", line 60, in <module>
    #     result.show(10, vertical=False, truncate=False)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/sql/dataframe.py", line 486, in show
    #     print(self._jdf.showString(n, int(truncate), vertical))
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/py4j/java_gateway.py", line 1304, in __call__
    #     return_value = get_return_value(
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/sql/utils.py", line 117, in deco
    #     raise converted from None
    # pyspark.sql.utils.PythonException:
    #   An exception was thrown from the Python worker. Please see the stack trace below.
    # Traceback (most recent call last):
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 604, in main
    #     process()
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 596, in process
    #     serializer.dump_stream(out_iter, outfile)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/python/lib/pyspark.zip/pyspark/sql/pandas/serializers.py", line 273, in dump_stream
    #     return ArrowStreamSerializer.dump_stream(self, init_stream_yield_batches(), stream)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/python/lib/pyspark.zip/pyspark/sql/pandas/serializers.py", line 81, in dump_stream
    #     for batch in iterator:
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/python/lib/pyspark.zip/pyspark/sql/pandas/serializers.py", line 266, in init_stream_yield_batches
    #     for series in iterator:
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 356, in func
    #     for result_batch, result_type in result_iter:
    #   File "/Users/renkaige/renkai-lab/rikai/python/rikai/spark/sql/codegen/spark.py", line 35, in spark_ml_udf
    #     model = spec.load_model()
    #   File "/Users/renkaige/renkai-lab/rikai/python/rikai/spark/sql/codegen/mlflow_registry.py", line 91, in load_model
    #     return getattr(mlflow, self.flavor).load_model(self.model_uri)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/mlflow/spark.py", line 627, in load_model
    #     return _load_model(model_uri=model_uri, dfs_tmpdir_base=dfs_tmpdir)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/mlflow/spark.py", line 578, in _load_model
    #     model_uri = _HadoopFileSystem.maybe_copy_from_uri(model_uri, dfs_tmpdir)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/mlflow/spark.py", line 354, in maybe_copy_from_uri
    #     return cls.maybe_copy_from_local_file(_download_artifact_from_uri(src_uri), dst_path)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/mlflow/spark.py", line 317, in maybe_copy_from_local_file
    #     local_path = cls._local_path(src)
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/mlflow/spark.py", line 291, in _local_path
    #     return cls._jvm().org.apache.hadoop.fs.Path(os.path.abspath(path))
    #   File "/Users/renkaige/miniconda/envs/rikai-example/lib/python3.8/site-packages/mlflow/spark.py", line 274, in _jvm
    #     return SparkContext._gateway.jvm
    # AttributeError: 'NoneType' object has no attribute 'jvm'
    result.show(10, vertical=False, truncate=False)

    # result = lrModel.transform(training)
    # result.printSchema()
    # result.show()
