from typing import List, Tuple
from shapely.geometry import Polygon
import numpy as np

def filter_by_aspect_ratio(df, aspect_ratio_col, config):
    """
    Classifies bounding boxes into 'normal' and 'irregular' based on aspect ratio.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box information.
        aspect_ratio_col (str): Column name for aspect ratios in the DataFrame.
        min_ratio (float): Minimum aspect ratio for a bounding box to be classified as 'normal'.
        max_ratio (float): Maximum aspect ratio for a bounding box to be classified as 'normal'.

    Returns:
        pd.DataFrame: Original DataFrame with a new 'type' column indicating 'normal' or 'irregular'.
        pd.DataFrame: Subset of the DataFrame with only 'normal' bounding boxes.
        pd.DataFrame: Subset of the DataFrame with only 'irregular' bounding boxes.
    """
    
    aspect_ratio_filter = config['ASPECT_RATIO_FILTER']
    min_ratio = aspect_ratio_filter['min_ratio']
    max_ratio = aspect_ratio_filter['max_ratio']
    # Classify bounding boxes
    df['type'] = df[aspect_ratio_col].apply(
        lambda ar: 'normal' if min_ratio <= ar <= max_ratio else 'irregular'
    )
    
    # Separate normal and irregular bounding boxes
    normal_bboxes = df[df['type'] == 'normal']
    irregular_bboxes = df[df['type'] == 'irregular']
    
    return df, normal_bboxes, irregular_bboxes

def filter_squares_by_area(df, aspect_ratio_col, area_col,config):
    """
    Filters bounding boxes based on aspect ratio and area thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box information.
        aspect_ratio_col (str): Column name for aspect ratios in the DataFrame.
        area_col (str): Column name for areas in the DataFrame.
        min_aspect_ratio (float): Minimum aspect ratio for filtering bounding boxes.
        max_area (float): Maximum area for filtering bounding boxes.

    Returns:
        pd.DataFrame: Subset of the DataFrame with bounding boxes that satisfy the conditions.
    """
    square_area_filter = config['AREA_FILTER_SQUARE']
    min_aspect_ratio = square_area_filter['min_aspect_ratio']
    max_area = square_area_filter['square_max_area']
    filtered_df = df[
        (df[aspect_ratio_col] > min_aspect_ratio) &
        (df[area_col] <= max_area)
    ].copy()
    return filtered_df

def filter_rects_by_area(df, aspect_ratio_col, area_col,config):
    # Second filter: Square filter
    rect_area_filter = config['AREA_FILTER_RECT']
    df_square_filtered = df[
        
            (df['aspect_ratio'] <= rect_area_filter['max_aspect_ratio']) &
            (df['area'] <= rect_area_filter['max_area'])
        
    ].copy()
    
    return df_square_filtered

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

def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    
    Args:
        boxes: numpy array of shape (N, 4, 2)
        scores: numpy array of shape (N,)    
    
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.

    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons, err_idexes = convert_format(boxes)

    top = 64
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1][:64]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def apply_nms_on_pseudo_labels(corners, nms_iou_threshold) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shape of pred is [N, 4, 2]
    
    Args: 
        pred: a numpy array of shape (N, 4, 2)
    Returns:
        corners: a numpy array of shape (N, 4, 2)
        scores: a numpy array of shape (N,)
    """
    scores = np.ones(corners.shape[0])
    selected_ids = non_max_suppression(corners, scores, nms_iou_threshold)
    corners = corners[selected_ids]
    scores = scores[selected_ids]
    return corners, selected_ids