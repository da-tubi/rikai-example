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

(
spark
    .read
    .format("parquet")
    .load("/tmp/rikai_example/elephants_dream")
    .createOrReplaceTempView("elephants_dream")
)

(
spark
    .sql("select frame_id, date_format(ts, 'mm:ss') from elephants_dream where frame_id < 100 order by frame_id asc limit 10")
    .show()
)
