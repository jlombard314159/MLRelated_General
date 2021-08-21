# ----- general utils
import torch
import selectivesearch
import numpy as np

# function to make tensor of shape (n,4) to hold n bounding boxes in [x1, y1, x2, y2] format
def xywh2xyxy(xywh):
    xyxy = xywh.new(xywh.shape)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return xyxy

# function to make tnesor of shape (n,4) to hold n bounding boxes in [xc, yc, w, h] format
def xyxyh2xywh(xyxy, img_size=516):
    xywh = torch.zeros(xyxy.shape[0], 6)
    xywh[:, 2] = (xyxy[:, 0] + xyxy[:, 2]) / 2. / img_size
    xywh[:, 3] = (xyxy[:, 1] + xyxy[:, 3]) / 2. / img_size
    xywh[:, 4] = (xyxy[:, 2] - xyxy[:, 0]) / img_size
    xywh[:, 5] = (xyxy[:, 3] - xyxy[:, 1]) / img_size
    xywh[:, 1] = xyxy[:, 6]
    return xywh

def extract_candidates(img):
    # ipdb.set_trace()
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=20)
    img_area = np.prod(img.shape[:2])
    candidates = []
    for r in regions:
        if r['rect'] in candidates: continue
        if r['size'] < (0.01*img_area): continue
        if r['size'] > (1*img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates

def extract_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou