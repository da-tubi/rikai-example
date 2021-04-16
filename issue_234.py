from pyspark.sql import Row

df = spark.createDataFrame([Row(id=1, split="train"), Row(id=2, split="eval")])
df.write.format("rikai").save("/tmp/issue_234")
df = spark.read.format("rikai").load("/tmp/rikai_example/issue_234")
df.registerTempTable("df")
spark.sql("SELECT * FROM df WHERE split = 'eval'").show()
