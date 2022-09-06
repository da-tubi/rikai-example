import os
import pathlib

from pyspark.sql.functions import col, lit
from rikai.types import Segment, VideoStream

from example import spark

video = VideoStream("elephants_dream.mp4")

df_video = spark.createDataFrame([(video, Segment(0, 14400))], ["video", "segment"])
df_video.createOrReplaceTempView("t_video")


sample_rate = 1
max_samples = 14400
tmp_dir = "/tmp/videostream_5"
if not pathlib.Path(tmp_dir).exists():
    os.mkdir(tmp_dir)

df_images = spark.sql(
    f"""
from (
    from (
        from t_video
        select video_to_images(video, "{tmp_dir}", segment, {sample_rate}, {max_samples}) as images
    )
    select explode(images) as image
)
select extract_uri(image) as uri
"""
)

(
    df_images.repartition(10)  # Control the number of files
    .write.format("rikai")
    .mode("overwrite")
    .save("/tmp/rikai_example/yolov5_video")
)
