import rikai
import pathlib
import subprocess
import torchvision
import torch

resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    progress=False,
)
model_uri = "/tmp/model/fasterrcnn_resnet50_fpn.pt"
torch.save(resnet, model_uri)

rikai.spark.sql.init(spark)

work_dir = pathlib.Path().absolute()

spark.sql(f"""
create model resnet_m
options (device="cpu")
using '{work_dir}/model/resnet_spec.yaml'
""")

result = spark.sql(f"""
select ML_PREDICT(resnet_m, '{work_dir}/img/lena.png')
""")

result.printSchema()
result.show()
