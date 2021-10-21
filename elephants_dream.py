from pyspark.sql import DataFrame, Row
from example import spark
import pickle
import cv2

from rikai.types.vision import Image

def load_video(spark, uri: str, limit = 400) -> DataFrame:
    cap = cv2.VideoCapture(uri)
    frame_id = 0
    rows = []
    while True:
        ret_val, img0 = cap.read()
        if not ret_val:
            print(f"last frame_id {frame_id}\n")
            break
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img_np = Image.from_array(img).to_numpy()
        row = Row(frame_id=frame_id, image=pickle.dumps(img_np, protocol=0))
        rows.append(row)
        if frame_id % 100 == 0:
            print(f"frame_id: {frame_id}\n")
        frame_id = frame_id + 1
        if frame_id >= limit:
            break
    cap.release()

    df = spark.createDataFrame(rows)
    return df

df = load_video(spark, "elephants_dream.mp4")
(
df
    .repartition(4)
    .write
    .format("rikai")
    .mode("overwrite")
    .save("/tmp/rikai_example/elephants_dream")
)
