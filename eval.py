from shapely.geometry import Polygon
import numpy as np
from typing import List, Tuple 

def convert_format(boxes_array: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    # boxes_array is a numpy array of shape (N, 4, 2)
    polygons = []
    err_idxs = []
    for idx, box in enumerate(boxes_array):
        try: 
            polygon = Polygon([(point[0], point[1]) for point in box] + [(box[0, 0], box[0, 1])])
            polygons.append(polygon)
        except Exception as e:
            print(f"Error converting bbox at index {idx}: {e}")
            err_idxs.append(idx)
                            
    return np.array(polygons), err_idxs

def compute_iou(box: Polygon, boxes: List[Polygon]):
    """Calculates IoU of the given box with the array of the given boxes.
    Note: the areas are passed in rather than calculated here for efficiency. 
    Calculate once in the caller to avoid duplicate work.
    
    Args:
        box: a polygon (shapely.geometry.Polygon)
        boxes: a numpy array of shape (N,), where each member is a shapely.geometry.Polygon
    Returns:
        a numpy array of shape (N,) containing IoU values
    """
    iou_lst = []
    for b in boxes:
        intersection = box.intersection(b).area
        union = box.union(b).area
        iou = intersection / union if union > 0 else 0
        iou_lst.append(iou)
    return np.array(iou_lst, dtype=np.float32)


def compute_overlaps(boxes1: np.ndarray, boxes2: np.ndarray):
    """Computes IoU overlaps between two sets of boxes.
    Returns an overlap matrix, which contains the IoU value for each combination of boxes.
    For better performance, pass the largest set first and the smaller second.
    
    Args: 
        boxes1: a numpy array of shape (N, 4, 2)
        boxes2: a numpy array of shape (M, 4, 2)
    Returns:
        overlaps: a numpy array of shape (N, M)
    """
    
    boxes1, _ = convert_format(boxes1)
    boxes2, _ = convert_format(boxes2)
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1)
    return overlaps


def compute_matches(gt_boxes: np.ndarray, #label_list
                    pred_boxes: np.ndarray, # corner_list
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    
    Args:
        gt_boxes: [N, 4, 2] Coordinates of ground truth boxes
        pred_boxes: [N, 4, 2] Coordinates of predicted boxes
        pred_scores: [N,] Confidence scores of predicted boxes
        iou_threshold: Float. IoU threshold to determine a match.
        score_threshold: Float. Score threshold to determine a match.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    if len(pred_boxes) == 0:
        return -1 * np.ones([gt_boxes.shape[0]]), np.array([]), np.array([])
    
    pred_scores = np.ones((len(pred_boxes),))

    gt_class_ids = np.ones(len(gt_boxes), dtype=int)
    pred_class_ids = np.ones(len(pred_scores), dtype=int)

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    overlaps = compute_overlaps(pred_boxes, gt_boxes)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0] #np.where returns a tuple (array, ) for 1D np array
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs: 
            # If ground truth box is already matched, go to next one
            if gt_match[j] > 0:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break #NOTE: sorted_ixs is in descending order, so if iou < iou_threshold, all the following ious will be less than iou_threshold
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps

def compute_ap(pred_match: np.ndarray, num_gt: int, num_pred: int):
    """ Compute Average Precision at a set IoU threshold (default 0.5).

        Args:
            pred_match: 1-D array. For each predicted box, it has the index of
                        the matched ground truth box.
            num_gt: Number of ground truth boxes
            num_pred: Number of predicted boxes
    """
    tp = (pred_match > -1).sum()
    precision = tp / num_pred
    recall = tp / num_gt
    return precision, recall

def compute_ap2(pred_match: np.ndarray, num_gt: int, num_pred: int) -> Tuple:
    """ Compute Average Precision at a set IoU threshold (default 0.5).

        Args:
            pred_match: 1-D array. For each predicted box, it has the index of
                        the matched ground truth box.
            num_gt: Number of ground truth boxes
            num_pred: Number of predicted boxes
            
        Returns:
            mAP: mean average precision
            precisions: precision values at each recall threshold
            recalls: recall values at each precision threshold
            precision: precision value
            recall: recall value
    """
    print(f"METHOD: compute_ap was called")
    assert num_gt != 0
    # assert num_pred != 0
    
    # Handle case when there are no predictions
    if num_pred == 0:
        print(f"METHOD: compute_ap: No predictions")
        # If there are no predictions, precision is 0 and recall depends on gt
        mAP = 0.0
        precisions = np.array([0,0])
        recalls = np.array([0, 1])  # Recall jumps from 0 to 1 over the recall range
        precision = 0.0
        recall = 0.0
        return mAP, precisions, recalls, precision, recall
    
    tp = (pred_match > -1).sum()
    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(num_pred) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / num_gt

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])
    precision = tp / num_pred
    recall = tp / num_gt
    return mAP, precisions, recalls, precision, recall
