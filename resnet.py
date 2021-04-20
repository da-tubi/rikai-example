import rikai
import pathlib
import subprocess

model_path = "/tmp/model/resnet50.pt"

# Download Coco Sample Dataset from Fast.ai datasets
if not os.path.exists(model_path):
    subprocess.check_call(f"wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O {model_path}", shell=True)
else:
    print("Model already downloaded...")

rikai.spark.sql.init(spark)

work_dir = pathlib.Path().absolute()

spark.sql(f"""
create model resnet_m
USING '{work_dir}/model/resnet_spec.yaml'
""")

spark.sql(f"""
select ML_PREDICT(resnet_m, '{work_dir}/img/lena.png')
""").show()
