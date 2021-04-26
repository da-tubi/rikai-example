from example import spark

spark.sql("show models").show(1, vertical=False, truncate=False)
