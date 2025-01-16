from typing import List, Tuple
from shapely.geometry import Polygon
import numpy as np

def apply_rect_filter(df, aspect_ratio_col, area_col, max_ratio, max_area):
    """
    Filters bounding boxes based on aspect ratio and area thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box information.
        aspect_ratio_col (str): Column name for aspect ratios in the DataFrame.
        area_col (str): Column name for areas in the DataFrame.
        max_ratio (float): Minimum aspect ratio for filtering bounding boxes.
        max_area (float): Minimum area for filtering bounding boxes.

    Returns:
        tuple: (narrow_boxes, other_boxes)
            - narrow_boxes (pd.DataFrame): Subset of the DataFrame containing bounding boxes classified as narrow.
            - other_boxes (pd.DataFrame): Subset of the DataFrame containing all other bounding boxes.
    """
    # Create a single mask for narrow rectangles
    narrow_mask = (df[aspect_ratio_col] < max_ratio) & (df[area_col] < max_area)

    # Apply the mask to separate narrow and other rectangles
    narrow_boxes = df[narrow_mask]
    other_boxes = df[~narrow_mask]  # Complement of the narrow mask

    # Reset the indices for both DataFrames
    narrow_boxes.reset_index(drop=True, inplace=True)
    other_boxes.reset_index(drop=True, inplace=True)

    return narrow_boxes, other_boxes


