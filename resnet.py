import rikai
import pathlib
import subprocess

model_path = "/tmp/model/resnet50.pt"

# Download pre-trained resnet model
# Here is where I found the pretrained model:
# https://github.com/pytorch/vision/blob/8fb5838ca916fd4ace080dae0357e7c307037bef/torchvision/models/detection/faster_rcnn.py#L291
if not os.path.exists(model_path):
    subprocess.check_call(f"wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth -O {model_path}", shell=True)
else:
    print("Model already downloaded...")

rikai.spark.sql.init(spark)

work_dir = pathlib.Path().absolute()

spark.sql(f"""
create model resnet_m
USING '{work_dir}/model/resnet_spec.yaml'
""")

result = spark.sql(f"""
select ML_PREDICT(resnet_m, '{work_dir}/img/lena.png')
""")

result.printSchema()
result.show()
