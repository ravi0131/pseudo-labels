import pandas as pd
from prototype_utils import bboxes_df_to_numpy_corners
from stats_filter import apply_nms_on_pseudo_labels
from stats_filter2 import apply_large_sq_filter, apply_rect_filter
from config import CONFIG
import os
from config import CONFIG
from typing import Dict

# called twice. once for iou_0.3 and once for iou_0.5
def apply_filters(bboxes_df: pd.DataFrame, config_rect_filter: Dict, config_sq_filter: Dict) -> pd.DataFrame:
    rect_filter_max_ratio = config_rect_filter['MAX_RATIO']
    rect_filter_max_area = config_rect_filter['MAX_AREA']
    _, other_boxes_df = apply_rect_filter(bboxes_df, 'aspect_ratio', 'area',rect_filter_max_ratio,rect_filter_max_area )
    
    sq_filter_min_ratio = config_sq_filter['MIN_RATIO']
    sq_filter_min_area = config_sq_filter['MIN_AREA']
    
    _ , rest_boxes_df = apply_large_sq_filter(other_boxes_df, 'aspect_ratio', 'area', sq_filter_min_ratio, sq_filter_min_area)
    
    return rest_boxes_df
def process_frame(input_frame_path: str, frame_id: str, scene_save_path: str, config: Dict):
    """
    Saves the filtered bboxes for the given frame at scene_save_path/frame_id/
    saves two files, one for iou_0.3 and one for iou_0.5 as iou_0.3_.feather and iou_0.5_.feather respectively
    Args: 
        input_frame_path: path to load the bboxes' dataframe for the given frame. 
        frame_id: id of the frame of interest
        scene_save_path: the path to save the processed frame. Frame will be saved as feather file at scene_save_path/frame_id.feather
                        ensure that this directory exists before calling this function
        
    """
    try:
        #filter bboxes
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
        filtered_df_03 = apply_filters(bboxes_df, rect_filter_config['IOU_THRESHOLD_0.3'], sq_filter_config['IOU_THRESHOLD_0.3'])
        filtered_df_05 = apply_filters(bboxes_df, rect_filter_config['IOU_THRESHOLD_0.5'], sq_filter_config['IOU_THRESHOLD_0.5'])
        
        #apply nms
        _, selected_idxes_03 = apply_nms_on_pseudo_labels(bboxes_df_to_numpy_corners(filtered_df_03), CONFIG['NMS_IOU_THRESHOLD'])
        filtered_df_03_nms = filtered_df_03.iloc[selected_idxes_03]
        filtered_df_03_nms.reset_index(drop=True, inplace=True)
        
        _, selected_idxes_05 = apply_nms_on_pseudo_labels(bboxes_df_to_numpy_corners(filtered_df_05), CONFIG['NMS_IOU_THRESHOLD'])
        filtered_df_05_nms = filtered_df_05.iloc[selected_idxes_05]
        filtered_df_05_nms.reset_index(drop=True, inplace=True)
        
        if not os.path.exists(scene_save_path):
            raise ValueError(f"Path {scene_save_path} does not exist. Please create the directory before calling this function")
        
        frame_save_path = os.path.join(scene_save_path,frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        
        filtered_df_03_nms.to_feather(os.path.join(frame_save_path, "iou_0.3_.feather"))
        filtered_df_05_nms.to_feather(os.path.join(frame_save_path, "iou_0.5_.feather"))
        
    except Exception as e:
        print(f"Error processing frame {frame_id} : {e}")
        
    
def process_scene(input_scene_path: str, scene_id: str, output_save_path: str, config: Dict):
    """
    Args: 
        scene_id: the id of the scene
        input_scene_path: the path to the scene  {os.listdir(input_scene_path) will return the frames}
        output_save_path: the base path to save each scene. Each frame will be saved as feather file at output_save_path/scene_id/frame_id.feather
    """
    scene_save_path = os.path.join(output_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)
    try:
        for input_frame in os.listdir(input_scene_path):
            input_frame_path = os.path.join(input_scene_path, input_frame)
            frame_id = input_frame.split(".")[0]
            process_frame(input_frame_path, frame_id, scene_save_path, config)
    except Exception as e:
        print(f"Error processing scene {scene_id} : {e}")
    
    # for input_frame in os.listdir(input_scene_path):
    #         input_frame_path = os.path.join(input_scene_path, input_frame)
    #         frame_id = input_frame.split(".")[0]
    #         process_frame(input_frame_path, frame_id, scene_save_path, config)
        
        
def process_all_scenes(input_path: str, output_save_path: str, config: Dict):
    """
    Args: 
        input_path: the path to the scenes {os.listdir(input_path) will return the scenes}
    """
    os.makedirs(output_save_path, exist_ok=True)
    for scene in os.listdir(input_path):
        input_scene_path = os.path.join(input_path, scene)
        process_scene(input_scene_path, scene, output_save_path, config)
    


def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    if CONFIG['ROI']:
        pl_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['ROI'])
        filtered_pl_save_path = os.path.join(home, *CONFIG['FILTERED_BBOX_FILE_PATHS']['ROI'])
    else:
        pl_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['FULL_RANGE'])
        filtered_pl_save_path = os.path.join(home, *CONFIG['FILTERED_BBOX_FILE_PATHS']['FULL_RANGE'])
    
    process_all_scenes(pl_path, filtered_pl_save_path, CONFIG) 
    print("Done")

if __name__ == "__main__":
    main()