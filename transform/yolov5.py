from typing import Any, Callable, Dict

from torchvision import transforms as T
import funcy

from rikai.contrib.torch.transforms.utils import uri_to_pil

__all__ = ["pre_processing"]

def pre_processing(options: Dict[str, Any]) -> Callable:
    # return T.Compose(
    #     [
    #         uri_to_pil,
    #         T.ToTensor(),
    #     ]
    # )
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
                "labels": [],
                "scores": [],
            }
            for *box, conf, cls in predicts.tolist():
                predict_result["boxes"].append(box)
                predict_result["labels"].append(cls)
                predict_result["scores"].append(conf)
            results.append(predict_result)
        return results

    return post_process_func
