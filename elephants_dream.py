from example import spark

df = spark.read.format("video").load("elephants_dream.mp4")
(
df
    .repartition(4)
    .write
    .format("rikai")
    .option("fps", 5)
    .mode("overwrite")
    .save("/tmp/rikai_example/elephants_dream")
)
