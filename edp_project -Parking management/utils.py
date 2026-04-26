# utils.py

import numpy as np

def midpoint(x1, y1, x2, y2):
    return (int((x1 + x2) // 2), int((y1 + y2) // 2))


def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h

    area1 = (bb_test[...,2]-bb_test[...,0])*(bb_test[...,3]-bb_test[...,1])
    area2 = (bb_gt[...,2]-bb_gt[...,0])*(bb_gt[...,3]-bb_gt[...,1])

    return inter / (area1 + area2 - inter)