import stats_filter
import pandas as pd
from prototype_utils import bboxes_df_to_numpy_corners
from config import CONFIG
import os
from config import CONFIG


def process_frame(input_frame_path: str, frame_id: str, scene_save_path: str):
    """
    Args: 
        input_frame_path: path to load the bboxes' dataframe for the given frame. 
        frame_id: id of the frame of interst
        scene_save_path: the path to save the processed frame. Frame will be saved as feather file at scene_save_path/frame_id.feather
                        ensure that this directory exists before calling this function
        
    """
    #filter bboxes
    bboxes_df = pd.read_feather(input_frame_path)
    bboxes_df['aspect_ratio'] = bboxes_df['box_width'] / bboxes_df['box_length']
    bboxes_df['area'] = bboxes_df['box_width'] * bboxes_df['box_length']
    _, normal_bboxes_df, _ = stats_filter.filter_by_aspect_ratio(bboxes_df, 'aspect_ratio', CONFIG)
    square_filter_df = stats_filter.filter_squares_by_area(normal_bboxes_df, 'aspect_ratio', 'area', CONFIG)
    rect_filter_df = stats_filter.filter_rects_by_area(normal_bboxes_df, 'aspect_ratio', 'area', CONFIG)

    combined_df = pd.concat([square_filter_df, rect_filter_df]).drop_duplicates()
    combined_df.reset_index(drop=True, inplace=True)
    #apply nms
    _, selected_idxes = stats_filter.apply_nms_on_pseudo_labels(bboxes_df_to_numpy_corners(combined_df), CONFIG['NMS_IOU_THRESHOLD'])
    nms_df = combined_df.iloc[selected_idxes]

    nms_df.reset_index(drop=True, inplace=True)
    if not os.path.exists(scene_save_path):
        raise ValueError(f"Path {scene_save_path} does not exist. Please create the directory before calling this function")
    nms_df.to_feather(os.path.join(scene_save_path, f"{frame_id}.feather"))
    
def process_scene(input_scene_path: str, scene_id: str, output_save_path: str):
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
            process_frame(input_frame_path, frame_id, scene_save_path)
            
    except Exception as e:
        print(f"Error processing scene {scene_id} : {e}")
        
        
def process_all_scenes(input_path: str, output_save_path: str):
    """
    Args: 
        input_path: the path to the scenes {os.listdir(input_path) will return the scenes}
    """
    os.makedirs(output_save_path, exist_ok=True)
    for scene in os.listdir(input_path):
        input_scene_path = os.path.join(input_path, scene)
        process_scene(input_scene_path, scene, output_save_path)
    


def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    if CONFIG['ROI']:
        pl_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['ROI'])
        filtered_pl_save_path = os.path.join(home, *CONFIG['FILTERED_BBOX_FILE_PATHS']['ROI'])
    else:
        pl_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['FULL_RANGE'])
        filtered_pl_save_path = os.path.join(home, *CONFIG['FILTERED_BBOX_FILE_PATHS']['FULL_RANGE'])
    
    process_all_scenes(pl_path, filtered_pl_save_path) 
    

if __name__ == "__main__":
    main()