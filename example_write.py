from pyspark.sql import Row
from pyspark.ml.linalg import DenseMatrix
from rikai.types import Image, Box2d
from rikai.numpy import wrap
import numpy as np
from example import spark

df = spark.createDataFrame(
    [
        {
            "id": 1,
            "mat": DenseMatrix(2, 2, range(4)),
            "image": Image("img/lena.png"),
            "annotations": [
                Row(
                    label="cat",
                    mask=wrap(np.random.rand(256, 256)),
                    bbox=Box2d(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                )
            ],
        }
    ]
)

df.write.format("rikai").save("/tmp/rikai_example/example_write")

spark.sql("select * from parquet.`/tmp/rikai_example/example_write`").show(1, vertical=True, truncate=False)
