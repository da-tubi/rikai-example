from pyspark.sql import DataFrame, Row
from example import spark
import pickle
import cv2
import numpy as np

from rikai.types.vision import Image

def load_video(spark, uri: str, limit = 1000, ratio = 0.99) -> DataFrame:
    cap = cv2.VideoCapture(uri)

    frame_id = 0
    rows = []
    prev_img = None
    while True:
        ret_val, img0 = cap.read()
        if not ret_val:
            print(f"last frame_id {frame_id}\n")
            break
        if prev_img is not None:
            bool_arr = (img0 == prev_img)
            true_ratio = np.count_nonzero(bool_arr) * 1.0 / bool_arr.size
            if true_ratio > ratio:
                print(f"true_ratio: {true_ratio} ratio: {ratio}")
                # The frames is almost the same, skip it
                frame_id = frame_id + 1
                continue
            else:
                prev_img = img0
        else:
            prev_img = img0

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

df = load_video(spark, "elephants_dream.mp4", ratio=0.75)
(
df
    .repartition(4)
    .write
    .format("rikai")
    .mode("overwrite")
    .save("/tmp/rikai_example/elephants_dream")
)
