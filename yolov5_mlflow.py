import getpass
import os
import urllib.request
import pathlib

import mlflow
import rikai
import yolov5
import torch
from rikai.contrib.torch.transforms.yolov5 import OUTPUT_SCHEMA

from example import spark

mlflow_tracking_uri = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(mlflow_tracking_uri)
registered_model_name = f"yolov5s-model"
mlflow.set_experiment(registered_model_name)

with mlflow.start_run():
    ####
    # Part 1: Train the model and register it on MLflow
    ####
    pretrained = 'yolov5s.pt'
    url = "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt"
    if os.path.exists(pretrained):
        urllib.request.urlretrieve(url, pretrained)
    model = yolov5.load(pretrained)
    pre = 'transform.yolov5.pre_processing'
    post = 'rikai.contrib.torch.transforms.yolov5.post_processing'

    # Rikai's logger adds output_schema, pre_pocessing, and post_processing as additional
    # arguments and automatically adds the flavor / rikai model spec version
    rikai.mlflow.pytorch.log_model(
        model,
        "model",
        OUTPUT_SCHEMA,
        pre_processing = pre,
        post_processing = post,
        registered_model_name = registered_model_name)

    ####
    # Part 2: create the model using the registered MLflow uri
    ####
    if torch.cuda.is_available():
        print("Using GPU\n")
        device = 'gpu'
    else:
        print("Using CPU\n")
        device = 'cpu'

    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri)
    spark.sql(f"""
    CREATE MODEL mlflow_yolov5_m OPTIONS (device='{device}') USING 'mlflow:///{registered_model_name}';
    """)

    ####
    # Part 3: predict using the registered Rikai model
    ####
    spark.sql("show models").show(1, vertical=False, truncate=False)

    work_dir = pathlib.Path().absolute()
    result = spark.sql(f"""
    select ML_PREDICT(mlflow_yolov5_m, '{work_dir}/img/lena.png')
    """)

    result.printSchema()
    result.show(1, vertical=False, truncate=False)
