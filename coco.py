import os
import subprocess
import pyspark.sql.functions as F
import json
from rikai.spark.functions import image, box2d_from_top_left
from pyspark.sql.functions import col, lit, concat, udf

coco_sample_path = "../coco_sample"

# Download Coco Sample Dataset from Fast.ai datasets
if not os.path.exists(coco_sample_path):
    subprocess.check_call("wget https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz -O - | tar -xz", shell=True)
else:
    print("Coco sample already downloaded...")

# Convert coco dataset into Rikai format
with open(f"{coco_sample_path}/annotations/train_sample.json") as fobj:
    coco = json.load(fobj)

categories_df = spark.createDataFrame(coco["categories"])

# Make sure that all bbox coordinates are float
anno_array = [{
    "image_id": a["image_id"],
    "bbox": [float(x) for x in a["bbox"]],
    "category_id": a["category_id"]
} for a in coco["annotations"]]

anno_df = (
    spark
    .createDataFrame(anno_array)
    .withColumn("box2d", box2d_from_top_left("bbox"))
)

# We could use JOIN to replace pycocotools.COCO
annotations_df = (
    anno_df.join(categories_df, anno_df.category_id == categories_df.id)
    .withColumn("anno", F.struct([col("box2d"), col("name"), col("category_id")]))
    .drop("box", "name", "id", "category_id")
    .groupBy(anno_df.image_id)
    .agg(F.collect_list("anno").alias("annotations"))
)

annotations_df.printSchema()
annotations_df.show(1, vertical=True, truncate=False)

## Build Coco dataset with image and annotations in Rikai format.
images_df = spark \
    .createDataFrame(spark.sparkContext.parallelize(coco["images"])) \
    .withColumn(
        "image", 
        image(concat(lit("coco_sample/train_sample/"), col("file_name")))
    )
images_df = images_df.join(annotations_df, images_df.id == annotations_df.image_id) \
    .drop("annotations_df.image_id", "file_name", "id")
images_df.show(1, vertical=True, truncate=False)
images_df.printSchema()

(
   images_df
   .repartition(4)  # Control the number of files
   .write
   .format("rikai")
   .mode("overwrite")
   .save("/tmp/rikai-example/coco")
)