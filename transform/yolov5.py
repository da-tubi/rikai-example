from typing import Any, Callable, Dict

from rikai.types.vision import Image
from yolov5.utils.datasets import exif_transpose, letterbox
from yolov5.utils.general import make_divisible, non_max_suppression, scale_coords
import numpy as np
import torch

__all__ = ["pre_processing", "post_processing", "OUTPUT_SCHEMA"]

def _pre_process_func(image_data):
    image_size = 640
    img = Image(image_data).to_pil()
    n, imgs = (1, [img])  # number of images, list of images
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames
    for i, im in enumerate(imgs):
        im = np.asarray(exif_transpose(im))
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = (image_size / max(s))  # gain
        shape1.append([y * g for y in s])
        imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    shape1 = [make_divisible(x, int(32)) for x in np.stack(shape1, 0).max(0)]  # inference shape
    x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
    x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
    x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
    x = torch.from_numpy(x).to("cpu").float() / 255.  # uint8 to fp16/32
    return x[0]


def pre_processing(options: Dict[str, Any]) -> Callable:
    return _pre_process_func


def post_processing(options: Dict[str, Any]) -> Callable:
    def post_process_func(batch: "Detections"):
        """
        Parameters
        ----------
        batch: Detections
            The ultralytics yolov5 (in torch hub) autoShape output
        """
        y = batch[0]
        y = non_max_suppression(y)  # NMS

        results = []

        # img = input
        # im0 = cv2.resize(open_cv_image, (640, 384))
        for i, det in enumerate(y):
            predict_result = {
                "boxes": [],
                "label_ids": [],
                "scores": [],
            }
            if det is not None and len(det):
                confs = np.around(det[:, 4].detach().cpu().numpy(), 2)
                # det[:, :4] = scale_coords(
                #     img.shape[2:], det[:, :4], im0.shape
                # ).round()
                det[:, :4] = scale_coords(
                    torch.Size([384, 640]), det[:, :4], (384, 640, 3)
                ).round()
        
                bbs = det[:, :4].detach().cpu().numpy().astype(int)
                for x, bb in enumerate(bbs):
                    predict_result["boxes"].append(bb)
                    predict_result["label_ids"].append(0)
                    predict_result["scores"].append(confs[x])
            results.append(predict_result)
        return results

        # results = []
        # for predicts in batch.pred:
        #     predict_result = {
        #         "boxes": [],
        #         "label_ids": [],
        #         "scores": [],
        #     }
        #     for *box, conf, cls in predicts.tolist():
        #         predict_result["boxes"].append(box)
        #         predict_result["label_ids"].append(cls)
        #         predict_result["scores"].append(conf)
        #     results.append(predict_result)
        # return results

    return post_process_func


OUTPUT_SCHEMA = (
    "struct<boxes:array<array<float>>, scores:array<float>, "
    "label_ids:array<int>>"
)
