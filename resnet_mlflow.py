import torchvision
import mlflow
import rikai
import getpass
import pathlib
from example import spark

mlflow_tracking_uri = "sqlite:///mlruns.db"

mlflow.set_tracking_uri(mlflow_tracking_uri)

with mlflow.start_run():
    ####
    # Part 1: Train the model and register it on MLflow
    ####

    # Using the pretrained here to simplify the example code
    resnet_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        progress=False,
    )

    schema = "struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>"
    pre = "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing"
    post = "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing"

    # Rikai's logger adds output_schema, pre_pocessing, and post_processing as additional
    # arguments and automatically adds the flavor / rikai model spec version
    registered_model_name = f"{getpass.getuser()}_resnet_model"
    rikai.mlflow.pytorch.log_model(
        resnet_model,
        "model",
        schema,
        pre,
        post,
        registered_model_name)

    ####
    # Part 2: create the model using the registered MLflow uri
    ####
    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri)
    spark.sql(f"""
    CREATE MODEL mlflow_resnet_m USING 'mlflow:///{registered_model_name}';
    """)

    ####
    # Part 3: predict using the registered Rikai model
    ####
    spark.sql("show models").show(1, vertical=False, truncate=False)

    work_dir = pathlib.Path().absolute()
    result = spark.sql(f"""
    select ML_PREDICT(mlflow_resnet_m, '{work_dir}/img/lena.png')
    """)

    result.printSchema()
    result.show(1, vertical=False, truncate=False)
