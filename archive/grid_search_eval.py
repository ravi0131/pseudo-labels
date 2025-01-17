from config import CONFIG
import pandas as pd
from typing import Dict, List, Tuple
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from prototype_utils import filter_cuboids_by_roi, extract_face_corners, filter_gt_labels_by_category, bboxes_df_to_numpy_corners
from eval_metrics import compute_matches, compute_ap2
from pathlib import Path
import os


def grid_search_compute_metrics_frame(input_frame_path: str,scene_id, frame_id: str,frame_save_path,av2: AV2SensorDataLoader, config: Dict):
    """
    Compute metrics for a single frame for a given filter and label type
    
    Args:
        input_frame_path : str : path to load pseudo labels
        scene_id : str : scene id
        frame_id : str : frame id
        frame_save_path : str : path to save metrics
        av2 : AV2SensorDataLoader : object to load ground truth
        config : Dict : configuration for the filter
        mode : Dict : mode to compute metrics for
        
    Returns:
        None
        
    Mode is a dictionary that controls which filter to compute metrics for
    positive means we want to read boxes that remain after the filter has been applied
    negative means that we want to read boxes that were filtered out by the filter
    
    input_frame_path should be the path to load pseudo labels(df)
    
    frame save path should be the directory where metrics are saved
    frame save path => /roi/ar_threshold_value/area_threshold_value/scene_id/frame_id
    
    """
    if config['ROI']:
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)
        filtered_cuboids = filter_cuboids_by_roi(relevant_cuboids.vertices_m, config)
        gt_corners = extract_face_corners(filtered_cuboids)
    else:
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id)).vertices_m
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)
        gt_corners = extract_face_corners(relevant_cuboids.vertices_m)
    
    mode = config['GRID_SEARCH_METRICS_MODE']
    #load pseudo_labels
    if mode['FILTER'] == 'rect_filter' and mode['LABEL_TYPE'] == 'positive':
        filename = "other_boxes.feather"
    elif mode['FILTER'] == 'rect_filter' and mode['LABEL_TYPE'] == 'negative':
        filename = "narrow_boxes.feather"
    elif mode['FILTER'] == 'square_filter' and mode['LABEL_TYPE'] == 'positive':
        filename = "rest_boxes.feather"
    elif mode['FILTER'] == 'square_filter' and mode['LABEL_TYPE'] == 'negative':
        filename = "large_squares.feather"
    else:
        raise ValueError(f"Invalid mode : {mode}")
        
    pseudo_labels_df = pd.read_feather(os.path.join(input_frame_path, filename))
    pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)

    #compute matches
    _ , pseudo_label_matches1, _  = compute_matches(gt_corners, pseudo_labels_corners, iou_threshold=0.3)
    _, pseudo_label_matches2, _ = compute_matches(gt_corners, pseudo_labels_corners, iou_threshold=0.5)
    
    gt_length = len(gt_corners)
    num_of_predictions = len(pseudo_labels_corners)
    
    mAP1, precisions1, recalls1, precision1, recall1 =compute_ap2(pseudo_label_matches1, gt_length, num_of_predictions)
    mAP2, precisions2, recalls2, precision2, recall2 =compute_ap2(pseudo_label_matches2, gt_length, num_of_predictions)
    
    
    save_dict1 = {
            "mAP": [mAP1],
            "precision": [precision1],
            "recall": [recall1],
            "precisions": [precisions1],
            "recalls": [recalls1],
            "pseudo_label_matches": [pseudo_label_matches1],
            "num_of_predictions": [num_of_predictions],
            "gt_length": [gt_length],
    }
    
    save_dict2 = {
            "mAP": [mAP2],
            "precision": [precision2],
            "recall": [recall2],
            "precisions": [precisions2],
            "recalls": [recalls2],
            "pseudo_label_matches": [pseudo_label_matches2],
            "num_of_predictions": [num_of_predictions],
            "gt_length": [gt_length],
    }
    
    results_df_1 = pd.DataFrame(save_dict1)
    results_df_2 = pd.DataFrame(save_dict2)

    if not os.path.exists(frame_save_path):
            raise ValueError(f"Path {frame_save_path} does not exist. Please create the directory before calling this function")
    
    results_df_1.to_feather(os.path.join(frame_save_path, f"iou_0.3_.feather"))
    results_df_2.to_feather(os.path.join(frame_save_path, f"iou_0.5_.feather"))

def grid_search_compute_metrics_scene(input_scene_path, scene_id, output_scene_path, av2: AV2SensorDataLoader, config: Dict):
    """
    Compute metrics for all frames in a single scene
    
    Args:
        input_scene_path : str : path to load pseudo labels
        scene_id : str : scene id
        output_scene_path : str : path to save metrics
        av2 : AV2SensorDataLoader : object to load ground truth
        config : Dict : configuration for the filter
    
    Returns:
        None
    """
    for frame_id in os.listdir(input_scene_path):
        try:
            input_frame_path = os.path.join(input_scene_path, frame_id)
            output_frame_path = os.path.join(output_scene_path, frame_id)
            os.makedirs(output_frame_path, exist_ok=True)
            grid_search_compute_metrics_frame(input_frame_path, scene_id,frame_id, output_frame_path,av2, config)
        except Exception as e:
            print(f"Error in frame {frame_id} and scene:{scene_id}: {e}")
            continue
    
    
def grid_search_compute_metrics_for_a_combination(base_scene_path: str, base_output_path:str, av2: AV2SensorDataLoader, config: Dict):
    """
    Compute metrics for all scenes for a given combination of aspect ratio and area threshold
    
    os.listdir(base_scene_path) should give you a list of scene ids
    Args:
        base_scene_path : str : path to load pseudo labels
        base_output_path : str : path to save metrics
        av2 : AV2SensorDataLoader : object to load ground truth
        config : Dict : configuration for the filter
    
    Returns:
        None
    """
    for scene_id in os.listdir(base_scene_path):
        input_scene_path = os.path.join(base_scene_path, scene_id)
        output_scene_path = os.path.join(base_output_path, scene_id)
        os.makedirs(output_scene_path, exist_ok=True)
        grid_search_compute_metrics_scene(input_scene_path,scene_id, output_scene_path, av2, config)
        
        

def grid_search_compute_metrics_for_filter(input_base_path: str, base_output_path: str, av2: AV2SensorDataLoader, config: Dict):
    """
    Compute metrics for a given filter (rect_filter or square_filter)
    
    os.listdir(input_base_path) should give you a list directories for aspect ratio threshold
    Args:
        input_base_path : str : path to load pseudo labels
        base_output_path : str : path to save metrics
        av2 : AV2SensorDataLoader : object to load ground truth
        config : Dict : configuration for the filter
    
    Returns:
        None
    """

    # compute metrics for a filter
    for ar_thres_dir in os.listdir(input_base_path):
        ar_threshold_value = ar_thres_dir.split("_")[-1]
        ar_threshold_value_path = os.path.join(input_base_path, ar_thres_dir)
        for area_thres_dir in os.listdir(ar_threshold_value_path):
            area_threshold_value = area_thres_dir.split("_")[-1]
            area_threshold_value_path = os.path.join(ar_threshold_value_path, area_thres_dir)
            combination_output_path = os.path.join(base_output_path, ar_thres_dir, area_thres_dir)
            grid_search_compute_metrics_for_a_combination(area_threshold_value_path, combination_output_path, av2, config)
            break
        break
        
def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    
    filter_type = CONFIG['GRID_SEARCH_METRICS_MODE']['FILTER']
    
    if CONFIG['ROI']:
        rect_filter_base_path = os.path.join(home, *CONFIG["GRID_SEARCH_RECT"]['PATH']['ROI'])
        if filter_type == 'rect_filter':
            grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['ROI']['RECT_FILTER_PATH'])
        else:
            grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['ROI']['SQUARE_FILTER_PATH'])
    else:
        rect_filter_base_path = os.path.join(home, *CONFIG["GRID_SEARCH_RECT"]['PATH']['FULL_RANGE'])
        if filter_type == 'rect_filter':
            grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['FULL_RANGE']['RECT_FILTER_PATH'])
        else:
            grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['FULL_RANGE']['SQUARE_FILTER_PATH'])

    av2_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))
    av2 = AV2SensorDataLoader(data_dir=av2_path, labels_dir=av2_path)

    grid_search_compute_metrics_for_filter(rect_filter_base_path, grid_search_metrics_output_path, av2, CONFIG)

if __name__ == '__main__':
    main()
