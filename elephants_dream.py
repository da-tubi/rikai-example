from example import spark

df = spark.read.format("video").load("elephants_dream.mp4")
from pyspark.sql.types import BinaryType
spark.udf.registerJavaFunction('image_data', "org.apache.spark.sql.rikai.expressions.ImageData", BinaryType())
df = df.selectExpr("image_data(image) as image_data", "frame_id")
(
df
    .repartition(4)
    .write
    .format("rikai")
    .mode("overwrite")
    .save("/tmp/rikai_example/elephants_dream")
)
