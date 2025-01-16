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


def apply_large_sq_filter(df, aspect_ratio_col, area_col, min_ratio, min_area):
    """
    Filters bounding boxes to identify large squares based on a single condition.

    Args:
        df (pd.DataFrame): DataFrame containing bounding box information.
        aspect_ratio_col (str): Column name for aspect ratios in the DataFrame.
        area_col (str): Column name for areas in the DataFrame.
        min_ratio (float): Minimum aspect ratio tolerance (ε).
        min_area (float): Minimum area for large squares (T_LargeArea).

    Returns:
        tuple: (large_squares, rest_boxes)
            - large_squares (pd.DataFrame): Subset of the DataFrame containing bounding boxes classified as large squares.
            - rest_boxes (pd.DataFrame): Subset of the DataFrame containing all other bounding boxes.
    """
    # Create a single mask for large squares
    large_square_mask = (df[aspect_ratio_col] >= min_ratio) & (df[aspect_ratio_col] <= 1.0) & (df[area_col] > min_area)

    # Apply the mask to separate large squares and other rectangles
    large_squares = df[large_square_mask]
    rest_boxes = df[~large_square_mask]  # Complement of the large square mask

    # Reset the indices for both DataFrames
    large_squares.reset_index(drop=True, inplace=True)
    rest_boxes.reset_index(drop=True, inplace=True)

    return large_squares, rest_boxes