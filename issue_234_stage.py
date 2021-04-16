from pyspark.sql import Row

df = spark.createDataFrame([Row(id=1, stage="train"), Row(id=2, stage="eval")])
df.write.format("rikai").save("/tmp/rikai_example/issue_234_stage")
df = spark.read.format("rikai").load("/tmp/rikai_example/issue_234_stage")
df.registerTempTable("df")
spark.sql("SELECT * FROM df WHERE stage = 'eval'").show()
