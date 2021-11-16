from typing import Any, Callable, Dict

from rikai.types.vision import Image
import funcy

__all__ = ["pre_processing", "post_processing", "OUTPUT_SCHEMA"]


def _pre_process_func(image_data):
    return funcy.identity # Image(image_data).to_pil()


def pre_processing(options: Dict[str, Any]) -> Callable:
    return funcy.identity


def post_processing(options: Dict[str, Any]) -> Callable:
    def post_process_func(batch: "Detections"):
        """
        Parameters
        ----------
        batch: Detections
            The ultralytics yolov5 (in torch hub) autoShape output
        """
        results = []
        for predicts in batch.pred:
            predict_result = {
                "boxes": [],
                "label_ids": [],
                "scores": [],
            }
            for *box, conf, cls in predicts.tolist():
                predict_result["boxes"].append(box)
                predict_result["label_ids"].append(cls)
                predict_result["scores"].append(conf)
            results.append(predict_result)
        return results

    return post_process_func


OUTPUT_SCHEMA = (
    "struct<boxes:array<array<float>>, scores:array<float>, "
    "label_ids:array<int>>"
)
