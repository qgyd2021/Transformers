#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
reference:
https://huggingface.co/spaces/nickmuchi/license-plate-detection-with-YOLOS
https://huggingface.co/docs/transformers/tasks/object_detection
"""
import argparse
import io
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import torch
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import validators

from project_settings import project_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default=(project_path / "pretrained_models/huggingface/hustvl/yolos-small").as_posix(),
        type=str
    )
    parser.add_argument(
        "--image_url_or_path",
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        type=str
    )
    parser.add_argument(
        "--threshold",
        default=0.25,
        type=float
    )
    # 0.5, 0.6, 0.7
    parser.add_argument("--iou_threshold", default=0.6, type=float)
    args = parser.parse_args()
    return args


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
]


def get_original_image(url_input):
    if validators.url(url_input):
        image = Image.open(requests.get(url_input, stream=True).raw)
        return image


def figure2image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_image = Image.open(buf)
    base_width = 750
    width_percent = base_width / float(pil_image.size[0])
    height_size = (float(pil_image.size[1]) * float(width_percent))
    height_size = int(height_size)
    pil_image = pil_image.resize((base_width, height_size), Image.Resampling.LANCZOS)
    return pil_image


def non_max_suppression(boxes, scores, threshold):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: array of [xmin, ymin, xmax, ymax]
        scores: array of scores associated with each box.
        threshold: IoU threshold
    Return:
        keep: indices of the boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # get boxes with more confidence first

    keep = []
    while order.size > 0:
        i = order[0]  # pick max confidence box
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def draw_boxes(image, boxes, scores, labels, threshold: float, idx_to_label: Dict[int, str] = None):
    plt.figure(figsize=(50, 50))
    plt.imshow(image)

    if idx_to_label is not None:
        labels = [idx_to_label[x] for x in labels]

    axis = plt.gca()
    colors = COLORS * len(boxes)
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        if score < threshold:
            continue
        axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=10))
        axis.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=60, bbox=dict(facecolor="yellow", alpha=0.8))
    plt.axis("off")

    return figure2image(plt.gcf())


def main():
    args = get_args()

    feature_extractor = YolosFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path)
    model = YolosForObjectDetection.from_pretrained(args.pretrained_model_name_or_path)

    # image
    image = get_original_image(args.image_url_or_path)
    image_size = torch.tensor([tuple(reversed(image.size))])

    # infer
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model.forward(**inputs)

    processed_outputs = feature_extractor.post_process(outputs, image_size)
    processed_outputs = processed_outputs[0]

    # draw box
    boxes = processed_outputs["boxes"].detach().numpy()
    scores = processed_outputs["scores"].detach().numpy()
    labels = processed_outputs["labels"].detach().numpy()

    keep = non_max_suppression(boxes, scores, threshold=args.iou_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    viz_image: Image = draw_boxes(
        image, boxes, scores, labels,
        threshold=args.threshold,
        idx_to_label=model.config.id2label
    )
    viz_image.show()

    return


if __name__ == '__main__':
    main()
