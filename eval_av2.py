from prototype_utils import filter_cuboids_by_roi, extract_face_corners, bboxes_df_to_numpy_corners
from eval_metrics import compute_matches, compute_ap2
from typing import Dict
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import os
import pandas as pd
from config import CONFIG


def calculate_metrics_frame(input_pl_frame_path: str, frame_id: str,scene_id: str, scene_save_path: str, av2:AV2SensorDataLoader,  config: Dict):
    try: 
        #load ground truth
        if config['ROI']: 
            cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id)).vertices_m
            filtered_cuboids = filter_cuboids_by_roi(cuboids, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id)).vertices_m
            gt_corners = extract_face_corners(cuboids)
        
        #load pseudo_labels
        pseudo_labels_df = pd.read_feather(input_pl_frame_path)
        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)
        # print(f"scene_save_path: {scene_save_path}")
        
        #compute matches
        _ , pseudo_label_matches, overlaps = compute_matches(gt_corners, pseudo_labels_corners)
        
        gt_length = len(gt_corners)
        num_of_predictions = len(pseudo_labels_corners)
        
        mAP, precisions, recalls, precision, recall =compute_ap2(pseudo_label_matches,gt_length,num_of_predictions)
        
        # print(f" type of mAP: {type(mAP)} \n type of precision: {type(precision)} \n type of recall: {type(recall)} \n type of precisions: {type(precisions)} \n type of recalls: {type(recalls)}")
        # print(f"precisions.shape: {precisions.shape} \nrecalls.shape: {recalls.shape}")
        results_df = pd.DataFrame({
            "mAP": [mAP],
            "precision": [precision],
            "recall": [recall],
            "precisions": [precisions],
            "recalls": [recalls],
            "pseudo_label_matches": [pseudo_label_matches],
            "num_of_predictions": [num_of_predictions],
            "gt_length": [gt_length],
        })
        
        if not os.path.exists(scene_save_path):
            raise ValueError(f"Path {scene_save_path} does not exist. Please create the directory before calling this function")
        
        frame_save_path = os.path.join(scene_save_path, f"{frame_id}.feather")
        results_df.to_feather(frame_save_path)    
    except Exception as e:
        print(f"shape of filtered_cuboids: {filtered_cuboids.shape} for scene_id: {scene_id} and frame_id: {frame_id}")
        print(f"Error processing frame {frame_id} in scene {scene_id} : {e}")
        
        
def calculate_metrics_scene(input_pl_path: str, scene_id: str, base_save_path: str, av2: AV2SensorDataLoader):
    scene_save_path = os.path.join(base_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)
    # count = 0
    for frame in os.listdir(input_pl_path): # frame is timsstamp.feather
        # count += 1
        input_frame_path = os.path.join(input_pl_path, frame)
        frame_id = frame.split(".")[0]
        # print(f"processing frame {frame}")
        # print(f"loading pseudo_labels from {input_frame_path}")
        # print(f"scene_save_path: {scene_save_path}")
        calculate_metrics_frame(input_frame_path, frame_id, scene_id, scene_save_path, av2, CONFIG)
        # if count == 1:
        #     break
    

def calculate_metrics_all_per_frame(input_pseudo_labels_path: str, av2: AV2SensorDataLoader, base_save_path):
    """
    Calculate the metrics for all the scenes
    """
    os.makedirs(base_save_path, exist_ok=True)
    # count = 0 
    for scene_id in os.listdir(input_pseudo_labels_path):
        # count += 1
        input_scene_path = os.path.join(input_pseudo_labels_path, scene_id)
        # print(f"loading pseudo_labels from {input_scene_path} for scne_id: {scene_id}" )
        calculate_metrics_scene(input_scene_path, scene_id, base_save_path, av2)
        # if count == 1:
        #     break


def calculate_total_metrics(base_save_path: str):
    """
    Calculate the total metrics for all the scenes
    
    Args:
        base_save_path: The base path where the scene folders of metrics are saved
    """
    mAPs = []
    precisions = []
    recalls = []
    for scene_id in os.listdir(base_save_path):
        scene_path = os.path.join(base_save_path, scene_id)
        for frame in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame)
            results_df = pd.read_feather(frame_path)
            mAPs.append(results_df['mAP'][0])
            precisions.append(results_df['precision'][0])
            recalls.append(results_df['recall'][0])
    mAP = sum(mAPs)/len(mAPs)
    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)
    return mAP, precision, recall


def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    dataset_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))
    av2 = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)
    
    if CONFIG['ROI']:
        normal_pl_input_path = os.path.join(home, *CONFIG["BBOX_FILE_PATHS"]['ROI'])
        filtered_pl_input_path = os.path.join(home, *CONFIG["FILTERED_BBOX_FILE_PATHS"]['ROI'])
        
        normal_pl_metrics_save_path = os.path.join(home, *CONFIG['METRICS_FILE_PATHS']['ROI']['NORMAL'])
        filtered_pl_metrics_save_path = os.path.join(home, *CONFIG['METRICS_FILE_PATHS']['ROI']['FILTERED'])
        
    
    else:
        normal_pl_input_path = os.path.join(home, *CONFIG["BBOX_FILE_PATHS"]['FULL_RANGE'])
        filtered_pl_input_path = os.path.join(home, *CONFIG["FILTERED_BBOX_FILE_PATHS"]['FULL_RANGE'])
        
        normal_pl_metrics_save_path = os.path.join(home, *CONFIG['METRICS_FILE_PATHS']['FULL_RANGE']['NORMAL'])
        filtered_pl_metrics_save_path = os.path.join(home, *CONFIG['METRICS_FILE_PATHS']['FULL_RANGE']['FILTERED'])
        


    calculate_metrics_all_per_frame(normal_pl_input_path, av2, normal_pl_metrics_save_path)
    calculate_metrics_all_per_frame(filtered_pl_input_path, av2, filtered_pl_metrics_save_path)
    
    mAP_normal, precision_normal, recall_normal = calculate_total_metrics(normal_pl_metrics_save_path)
    mAP_filtered, precision_filtered, recall_filtered = calculate_total_metrics(filtered_pl_metrics_save_path)
    
    print(f"mAP_normal: {mAP_normal}, precision_normal: {precision_normal}, recall_normal: {recall_normal}")
    print(f"mAP_filtered: {mAP_filtered}, precision_filtered: {precision_filtered}, recall_filtered: {recall_filtered}")

if __name__ == '__main__':
    main()