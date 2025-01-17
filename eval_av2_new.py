from prototype_utils import filter_cuboids_by_roi, extract_face_corners, bboxes_df_to_numpy_corners, filter_gt_labels_by_category
from eval_metrics import compute_matches, compute_ap2
from typing import Dict
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import os
import pandas as pd
from config import CONFIG


def calculate_metrics_frame_normal(input_pl_frame_path: str, frame_id: str,scene_id: str, scene_save_path: str, av2:AV2SensorDataLoader,  config: Dict, iou_threshold: float):
    """
    Calculate the metrics for a single frame (normal not pseudo-labels)
    Saves the metrics dataframe at scene_save_path/frame_id/iou_{iou_threshold}_.feather
    
    Args:
        input_pl_frame_path: Path to the pseudo-labels dataframe for the frame
        frame_id: The frame id
        scene_id: The scene id
        scene_save_path: The directory to save the metrics dataframe
        av2: AV2SensorDataLoader object
        config: The configuration dictionary
        iou_threshold: The iou_threshold to use for computing the metrics. Should be either "0.3" or "0.5"
        
    """
    try: 
        #load ground truth
        # if config['ROI']: 
        #     cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        #     relevant_cuboids = filter_gt_labels_by_category(cuboids, config)
        #     filtered_cuboids = filter_cuboids_by_roi(cuboids, config)
        #     gt_corners = extract_face_corners(filtered_cuboids)
        # else:
        #     cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id)).vertices_m
        #     gt_corners = extract_face_corners(cuboids)
        
        # 1) Grab cuboids
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        # 2) Filter by category
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)
        # 3) If ROI is on, filter by ROI; else skip
        if config["ROI"]:
            filtered_cuboids = filter_cuboids_by_roi(relevant_cuboids.vertices_m, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            gt_corners = extract_face_corners(relevant_cuboids.vertices_m)
        
        #load unfiltered pseudo_labels
        pseudo_labels_df = pd.read_feather(input_pl_frame_path)
        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)
        # print(f"scene_save_path: {scene_save_path}")
        
        if iou_threshold != 0.3 and iou_threshold != 0.5:
            raise ValueError(f"iou_threshold should be either 0.3 or 0.5. Got {iou_threshold} as type {type(iou_threshold)}.")

        #compute matches
        _ , pseudo_label_matches, overlaps = compute_matches(gt_corners, pseudo_labels_corners, iou_threshold=iou_threshold)
        
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
        
        # frame_save_path = os.path.join(scene_save_path, f"{frame_id}.feather")
        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        results_df.to_feather(os.path.join(frame_save_path, f"iou_{iou_threshold}_.feather"))    
    except Exception as e:
        print(f"NORMAL FRAME: shape of filtered_cuboids: {filtered_cuboids.shape} for scene_id: {scene_id} and frame_id: {frame_id}")
        print(f"NORMAL FRAME: Error processing frame {frame_id} in scene {scene_id} : {e}")



def calculate_metrics_frame_filtered(input_pl_frame_path: str, frame_id: str,scene_id: str, scene_save_path: str, av2:AV2SensorDataLoader,  config: Dict, iou_threshold):
    try: 
        # 1) Grab cuboids
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        # 2) Filter by category
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)
        # 3) If ROI is on, filter by ROI; else skip
        if config["ROI"]:
            filtered_cuboids = filter_cuboids_by_roi(relevant_cuboids.vertices_m, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            gt_corners = extract_face_corners(relevant_cuboids.vertices_m)
        
        #load filtered pseudo_labels
        if iou_threshold == 0.3:
            pseudo_labels_df = pd.read_feather(os.path.join(input_pl_frame_path, "iou_0.3_.feather"))
        elif iou_threshold == 0.5:
            pseudo_labels_df = pd.read_feather(os.path.join(input_pl_frame_path, "iou_0.5_.feather"))
        else:
            raise ValueError(f"iou_threshold should be either 0.3 or 0.5. Got {iou_threshold} with type {type(iou_threshold)}")
        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)
        # print(f"scene_save_path: {scene_save_path}")
        
        #compute matches
        _ , pseudo_label_matches, overlaps = compute_matches(gt_corners, pseudo_labels_corners, iou_threshold=float(iou_threshold))
        
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
        
        # frame_save_path = os.path.join(scene_save_path, f"{frame_id}.feather")
        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        results_df.to_feather(os.path.join(frame_save_path, f"iou_{iou_threshold}_.feather"))    
    except Exception as e:
        print(f"FILTERED FRAME: shape of filtered_cuboids: {filtered_cuboids.shape} for scene_id:")
        print(f"FILTERED FRAME: Error processing frame {frame_id} in scene {scene_id} : {e}")
    
def calculate_metrics_scene_normal(input_pl_path: str, scene_id: str, base_save_path: str, av2: AV2SensorDataLoader, iou_threshold):
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
        calculate_metrics_frame_normal(input_frame_path, frame_id, scene_id, scene_save_path, av2, CONFIG, iou_threshold)


def calculate_metrics_scene_filtered(input_pl_path: str, scene_id: str, base_save_path: str, av2: AV2SensorDataLoader, iou_threshold):
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
        calculate_metrics_frame_filtered(input_frame_path, frame_id, scene_id, scene_save_path, av2, CONFIG, iou_threshold)
        

def calculate_all_metrics_framewise_normal(input_pseudo_labels_path: str, av2: AV2SensorDataLoader, base_save_path, iou_threshold):
    """
    Calculate the metrics for all the scenes
    """
    os.makedirs(base_save_path, exist_ok=True)
    # count = 0 
    for scene_id in os.listdir(input_pseudo_labels_path):
        input_scene_path = os.path.join(input_pseudo_labels_path, scene_id)
        calculate_metrics_scene_normal(input_scene_path, scene_id, base_save_path, av2, iou_threshold)
        
        
def calculate_all_metrics_framewise_filtered(input_pseudo_labels_path: str, av2: AV2SensorDataLoader, base_save_path, iou_threshold):
    """
    Calculate the metrics for all the scenes
    """
    os.makedirs(base_save_path, exist_ok=True)
    # count = 0 
    for scene_id in os.listdir(input_pseudo_labels_path):
        input_scene_path = os.path.join(input_pseudo_labels_path, scene_id)
        calculate_metrics_scene_filtered(input_scene_path, scene_id, base_save_path, av2, iou_threshold)
        


def calculate_total_metrics_normal(base_save_path: str, iou_threshold: float):
    """
    Calculate the total metrics for all the scenes
    
    Args:
        base_save_path: The base path where the scene folders of metrics are saved
    """
    mAPs = []
    precisions = []
    recalls = []
    if str(iou_threshold) == "0.3":
        df_filename = "iou_0.3_.feather"
    elif str(iou_threshold) == "0.5":
        df_filename = "iou_0.5_.feather"
    else:
        raise ValueError(f"iou_threshold should be either 0.3 or 0.5. Got {iou_threshold} with type {type(iou_threshold)}")
    
    
    for scene_id in os.listdir(base_save_path):
        scene_path = os.path.join(base_save_path, scene_id)
        for frame in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame)
            results_df = pd.read_feather(os.path.join(frame_path, df_filename))
            mAPs.append(results_df['mAP'][0])
            precisions.append(results_df['precision'][0])
            recalls.append(results_df['recall'][0])
    mAP = sum(mAPs)/len(mAPs)
    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)
    return mAP, precision, recall


def main(iou_mode: float, av2: AV2SensorDataLoader):
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    
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
        


    calculate_all_metrics_framewise_normal(normal_pl_input_path, av2, normal_pl_metrics_save_path,iou_threshold=iou_mode)
    calculate_all_metrics_framewise_filtered(filtered_pl_input_path, av2, filtered_pl_metrics_save_path, iou_threshold=iou_mode)
    
    mAP_normal, precision_normal, recall_normal = calculate_total_metrics_normal(normal_pl_metrics_save_path, iou_threshold=iou_mode)
    mAP_filtered, precision_filtered, recall_filtered = calculate_total_metrics_normal(filtered_pl_metrics_save_path, iou_threshold=iou_mode)
    
    normal_metrics = {
        "mAP": mAP_normal,
        "precision": precision_normal,
        "recall": recall_normal
    }
    
    filtered_metrics = {
        "mAP": mAP_filtered,
        "precision": precision_filtered,
        "recall": recall_filtered
    }
    
    return normal_metrics, filtered_metrics

if __name__ == '__main__':
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    dataset_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))
    av2 = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)
    
    
    normal_metrics_03, filtered_metrics_03 = main(0.3, av2)
    normal_metrics_05, filtered_metrics_05 = main(0.5, av2)
    
    print("\n--- Results for IoU = 0.3 ---\n")
    print(f"Normal Pseudo-labels: mAP = {normal_metrics_03['mAP']}, Precision = {normal_metrics_03['precision']}, Recall = {normal_metrics_03['recall']}")
    print(f"Filtered Pseudo-labels: mAP = {filtered_metrics_03['mAP']}, Precision = {filtered_metrics_03['precision']}, Recall = {filtered_metrics_03['recall']}")
    
    print("\n--- Results for IoU = 0.5 ---\n")
    print(f"Normal Pseudo-labels: mAP = {normal_metrics_05['mAP']}, Precision = {normal_metrics_05['precision']}, Recall = {normal_metrics_05['recall']}")
    print(f"Filtered Pseudo-labels: mAP = {filtered_metrics_05['mAP']}, Precision = {filtered_metrics_05['precision']}, Recall = {filtered_metrics_05['recall']}")