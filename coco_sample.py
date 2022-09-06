import json
import os
import subprocess

import pyspark.sql.functions as F
from pyspark.sql.functions import col, concat, lit, udf
from rikai.spark.functions import box2d_from_top_left, to_image

from example import spark

coco_sample_path = "data/coco_sample"

# Download Coco Sample Dataset from Fast.ai datasets
if not os.path.exists(coco_sample_path):
    subprocess.check_call(
        "wget https://s3.amazonaws.com/fast-ai-coco/coco_sample.tgz -O - | tar -xz",
        shell=True,
    )
else:
    print("Coco sample already downloaded...")

# Convert coco dataset into Rikai format
with open(f"{coco_sample_path}/annotations/train_sample.json") as fobj:
    coco = json.load(fobj)

categories_df = spark.createDataFrame(coco["categories"])

# Make sure that all bbox coordinates are float
anno_array = [
    {
        "image_id": a["image_id"],
        "bbox": [float(x) for x in a["bbox"]],
        "category_id": a["category_id"],
    }
    for a in coco["annotations"]
]

anno_df = spark.createDataFrame(anno_array).withColumn(
    "box2d", box2d_from_top_left("bbox")
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
images_df = (
    spark.createDataFrame(spark.sparkContext.parallelize(coco["images"]))
    .withColumn(
        "image_path", concat(lit("data/coco_sample/train_sample/"), col("file_name"))
    )
    .withColumn(
        "image",
        to_image(concat(lit("data/coco_sample/train_sample/"), col("file_name"))),
    )
)
images_df = images_df.join(
    annotations_df, images_df.id == annotations_df.image_id
).drop("annotations_df.image_id", "file_name", "id")
images_df.show(1, vertical=True, truncate=False)
images_df.printSchema()

(
    images_df.repartition(4)  # Control the number of files
    .write.format("rikai")
    .mode("overwrite")
    .save("/tmp/rikai_example/coco_sample")
)

labels_text = """person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
street sign
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
hat
backpack
umbrella
shoe
eye glasses
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
plate
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
mirror
dining table
window
desk
toilet
door
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
blender
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
hair brush"""

(
    spark.createDataFrame(
        [row for row in enumerate(labels_text.split("\n"))],
        "id: integer, label: string",
    )
    .repartition(4)
    .write.format("rikai")
    .mode("overwrite")
    .save("/tmp/rikai_example/coco_labels")
)
