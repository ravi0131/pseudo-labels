import os
import pandas as pd
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import CONFIG
from prototype_utils import bboxes_df_to_numpy_corners
from stats_filter import apply_nms_on_pseudo_labels
from stats_filter2 import apply_large_sq_filter, apply_rect_filter


# -----------------------------------------------------------------------------
# 1) Filter application (unchanged)
# -----------------------------------------------------------------------------
def apply_filters(bboxes_df: pd.DataFrame, config_rect_filter: Dict, config_sq_filter: Dict) -> pd.DataFrame:
    rect_filter_max_ratio = config_rect_filter['MAX_RATIO']
    rect_filter_max_area = config_rect_filter['MAX_AREA']
    _, other_boxes_df = apply_rect_filter(
        bboxes_df, 'aspect_ratio', 'area', 
        rect_filter_max_ratio, rect_filter_max_area
    )
    
    sq_filter_min_ratio = config_sq_filter['MIN_RATIO']
    sq_filter_min_area = config_sq_filter['MIN_AREA']
    _, rest_boxes_df = apply_large_sq_filter(
        other_boxes_df, 'aspect_ratio', 'area', 
        sq_filter_min_ratio, sq_filter_min_area
    )
    
    return rest_boxes_df


# -----------------------------------------------------------------------------
# 2) Process a single frame (unchanged)
# -----------------------------------------------------------------------------
def process_frame(
    input_frame_path: str,
    frame_id: str,
    scene_save_path: str,
    config: Dict
):
    """
    Saves the filtered bboxes for the given frame at:
       scene_save_path/frame_id/iou_0.3_.feather
       scene_save_path/frame_id/iou_0.5_.feather
    """
    try:
        bboxes_df = pd.read_feather(input_frame_path)
        bboxes_df['aspect_ratio'] = bboxes_df['box_width'] / bboxes_df['box_length']
        bboxes_df['area'] = bboxes_df['box_width'] * bboxes_df['box_length']
        
        if config['ROI']:
            rect_filter_config = config['RECT_FILTER_THRESHOLDS']['ROI']    
            sq_filter_config = config['SQUARE_FILTER_THRESHOLDS']['ROI']
        else:
            rect_filter_config = config['RECT_FILTER_THRESHOLDS']['FULL_RANGE']
            sq_filter_config = config['SQUARE_FILTER_THRESHOLDS']['FULL_RANGE']
            
        # apply both filters
        filtered_df_03 = apply_filters(
            bboxes_df,
            rect_filter_config['IOU_THRESHOLD_0.3'],
            sq_filter_config['IOU_THRESHOLD_0.3']
        )
        filtered_df_05 = apply_filters(
            bboxes_df,
            rect_filter_config['IOU_THRESHOLD_0.5'],
            sq_filter_config['IOU_THRESHOLD_0.5']
        )
        
        # apply NMS
        _, selected_idxes_03 = apply_nms_on_pseudo_labels(
            bboxes_df_to_numpy_corners(filtered_df_03),
            CONFIG['NMS_IOU_THRESHOLD']
        )
        filtered_df_03_nms = filtered_df_03.iloc[selected_idxes_03].reset_index(drop=True)
        
        _, selected_idxes_05 = apply_nms_on_pseudo_labels(
            bboxes_df_to_numpy_corners(filtered_df_05),
            CONFIG['NMS_IOU_THRESHOLD']
        )
        filtered_df_05_nms = filtered_df_05.iloc[selected_idxes_05].reset_index(drop=True)
        
        if not os.path.exists(scene_save_path):
            raise ValueError(
                f"Path {scene_save_path} does not exist. "
                "Please create the directory before calling this function"
            )
        
        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        
        filtered_df_03_nms.to_feather(os.path.join(frame_save_path, "iou_0.3_.feather"))
        filtered_df_05_nms.to_feather(os.path.join(frame_save_path, "iou_0.5_.feather"))
        
    except Exception as e:
        print(f"Error processing frame {frame_id} : {e}")


# -----------------------------------------------------------------------------
# 3) Process a single scene (unchanged)
# -----------------------------------------------------------------------------
def process_scene(
    input_scene_path: str,
    scene_id: str,
    output_save_path: str,
    config: Dict
):
    """
    Processes all frames within a single scene.
    """
    scene_save_path = os.path.join(output_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)
    try:
        for input_frame in os.listdir(input_scene_path):
            input_frame_path = os.path.join(input_scene_path, input_frame)
            frame_id = os.path.splitext(input_frame)[0]
            process_frame(input_frame_path, frame_id, scene_save_path, config)

    except Exception as e:
        print(f"Error processing scene {scene_id} : {e}")


# -----------------------------------------------------------------------------
# 4) Process *all* scenes in parallel
# -----------------------------------------------------------------------------
def process_all_scenes(
    input_path: str,
    output_save_path: str,
    config: Dict
):
    """
    Parallelizes the processing of each scene.
    """
    os.makedirs(output_save_path, exist_ok=True)

    # We'll accumulate tasks in a list
    futures_map = {}
    with ProcessPoolExecutor() as executor:
        for scene in os.listdir(input_path):
            input_scene_path = os.path.join(input_path, scene)
            if not os.path.isdir(input_scene_path):
                continue

            # Submit a process_scene job for each scene
            future = executor.submit(process_scene, input_scene_path, scene, output_save_path, config)
            futures_map[future] = scene

        # Optionally, wait for them to complete & catch errors
        for future in as_completed(futures_map):
            scene_id = futures_map[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error in scene {scene_id}: {e}")


# -----------------------------------------------------------------------------
# 5) Main (unchanged except for calling the parallel process_all_scenes)
# -----------------------------------------------------------------------------
def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    if CONFIG['ROI']:
        pl_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['ROI'])
        filtered_pl_save_path = os.path.join(home, *CONFIG['FILTERED_BBOX_FILE_PATHS']['ROI'])
    else:
        pl_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['FULL_RANGE'])
        filtered_pl_save_path = os.path.join(home, *CONFIG['FILTERED_BBOX_FILE_PATHS']['FULL_RANGE'])
    
    # Use the parallel version of "process_all_scenes"
    process_all_scenes(pl_path, filtered_pl_save_path, CONFIG) 
    print("Done")


if __name__ == "__main__":
    main()
