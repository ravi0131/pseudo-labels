import os
import logging
import time
from functools import partial
from multiprocessing import Pool, cpu_count

from pathlib import Path
import pandas as pd
from typing import Dict

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader

from prototype_utils import filter_cuboids_by_roi, extract_face_corners, bboxes_df_to_numpy_corners
from eval_metrics import compute_matches, compute_ap2
from config import CONFIG


def get_main_logger(script_name: str) -> logging.Logger:
    """
    Creates a master logger that logs high-level information (success/failure of scenes, total time).
    """
    logs_dir = f"{script_name}_logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "master.log")
    
    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)
    
    # If logger has no handlers, add one; otherwise we might add duplicate handlers on re-runs.
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_scene_logger(scene_name: str, logs_dir: str) -> logging.Logger:
    """
    Creates a dedicated logger for each scene, storing logs in logs_dir/scene_name.log.
    """
    logger = logging.getLogger(f"scene_logger_{scene_name}")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        log_path = os.path.join(logs_dir, f"{scene_name}.log")
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_metrics_frame(
    input_pl_frame_path: str, 
    frame_id: str,
    scene_id: str, 
    scene_save_path: str, 
    av2: AV2SensorDataLoader,  
    config: Dict,
    scene_logger: logging.Logger
):
    """
    Calculate metrics for a single frame. 
    """
    try:
        # Load ground truth
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id)).vertices_m
        if config['ROI']:
            filtered_cuboids = filter_cuboids_by_roi(cuboids, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            gt_corners = extract_face_corners(cuboids)
        
        # Load pseudo labels
        pseudo_labels_df = pd.read_feather(input_pl_frame_path)
        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)
        
        # Compute matches
        _, pseudo_label_matches, overlaps = compute_matches(gt_corners, pseudo_labels_corners)
        gt_length = len(gt_corners)
        num_of_predictions = len(pseudo_labels_corners)
        
        mAP, precisions, recalls, precision, recall = compute_ap2(
            pseudo_label_matches, gt_length, num_of_predictions
        )
        
        # Save results to disk
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

        scene_logger.info(f"Frame {frame_id} -> "
                          f"mAP: {mAP:.3f}, "
                          f"precision: {precision:.3f}, "
                          f"recall: {recall:.3f}.")

    except Exception as e:
        scene_logger.error(
            f"Error processing frame {frame_id} in scene {scene_id}: {str(e)}"
        )


def process_frame(
    frame: str, 
    input_scene_path: str, 
    scene_id: str, 
    scene_save_path: str, 
    av2: AV2SensorDataLoader,  
    config: Dict,
    scene_logger: logging.Logger
):
    """
    Wrapper function for processing a single frame (pseudo-label + GT).
    """
    try:
        frame_id = frame.split(".")[0]
        input_frame_path = os.path.join(input_scene_path, frame)
        calculate_metrics_frame(
            input_frame_path, 
            frame_id, 
            scene_id, 
            scene_save_path, 
            av2, 
            config,
            scene_logger
        )
        return frame_id
    except Exception as e:
        # We log errors within `calculate_metrics_frame` as well.
        return None


def process_scene(scene_tuple):
    """
    Process a single scene in a single worker.
    """
    (
        scene_id,
        input_pseudo_labels_path, 
        base_save_path, 
        av2_path_str,
        config,
        logs_dir,
        script_name
    ) = scene_tuple
    
    scene_logger = get_scene_logger(scene_id, logs_dir)
    
    # Re-instantiate AV2SensorDataLoader in each process if needed 
    # (if it is not pickle-able or if you have issues with parallel usage):
    av2 = AV2SensorDataLoader(data_dir=Path(av2_path_str), labels_dir=Path(av2_path_str))
    
    try:
        scene_logger.info(f"Starting scene: {scene_id}")
        
        input_scene_path = os.path.join(input_pseudo_labels_path, scene_id)
        scene_save_path = os.path.join(base_save_path, scene_id)
        os.makedirs(scene_save_path, exist_ok=True)

        frames_processed = 0
        for frame in os.listdir(input_scene_path):
            result = process_frame(
                frame,
                input_scene_path,
                scene_id,
                scene_save_path,
                av2,
                config,
                scene_logger
            )
            if result is not None:
                frames_processed += 1
        
        scene_logger.info(f"Completed scene {scene_id}. Total frames processed: {frames_processed}")
        return (scene_id, True, frames_processed)
    
    except Exception as e:
        scene_logger.error(f"Error processing scene {scene_id}: {str(e)}")
        return (scene_id, False, 0)


def calculate_total_metrics(base_save_path: str):
    """
    Calculate the total metrics for all the scenes by aggregating per-frame .feather files.
    """
    mAPs = []
    precisions = []
    recalls = []
    
    for scene_id in os.listdir(base_save_path):
        scene_path = os.path.join(base_save_path, scene_id)
        # If not a directory or empty, skip
        if not os.path.isdir(scene_path) or len(os.listdir(scene_path)) == 0:
            continue
        
        for frame in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame)
            results_df = pd.read_feather(frame_path)
            mAPs.append(results_df['mAP'][0])
            precisions.append(results_df['precision'][0])
            recalls.append(results_df['recall'][0])
    
    # Handle edge case: no frames processed
    if len(mAPs) == 0:
        return 0.0, 0.0, 0.0
    
    mAP = sum(mAPs) / len(mAPs)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return mAP, precision, recall


def main():
    # Derive script name
    script_name = os.path.splitext(os.path.basename(__file__))[0] \
        if "__file__" in globals() else "metrics_calculation"

    # Create logs directory & get master logger
    logs_dir = f"{script_name}_logs"
    os.makedirs(logs_dir, exist_ok=True)
    master_logger = get_main_logger(script_name)

    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    av2_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))

    # We define the input paths and output paths
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

    os.makedirs(normal_pl_metrics_save_path, exist_ok=True)
    os.makedirs(filtered_pl_metrics_save_path, exist_ok=True)

    # We will process each of these input paths ( normal vs filtered ) in two separate parallel calls.

    def create_scene_tuples(input_path, output_path):
        """
        Create a list of scene tuples (one per scene), to be processed in parallel.
        (scene_id, input_pseudo_labels_path, base_save_path, av2_path_str, config, logs_dir, script_name)
        """
        scene_ids = os.listdir(input_path)
        # Optionally filter or limit the scenes:
        scene_count = CONFIG.get('SCENE_COUNT', len(scene_ids))  # Or fallback to all scenes
        scene_ids = scene_ids[:scene_count]

        return [
            (
                scene_id,
                input_path, 
                output_path, 
                str(av2_path),
                CONFIG,
                logs_dir,
                script_name
            )
            for scene_id in scene_ids
        ]
    
    # Parallel processing
    start_time = time.time()

    # 1) Process normal pseudo-labels
    normal_scene_tuples = create_scene_tuples(normal_pl_input_path, normal_pl_metrics_save_path)
    with Pool(processes=max(1, cpu_count() - 1), maxtasksperchild=1) as pool:
        normal_results = pool.map(process_scene, normal_scene_tuples)

    # 2) Process filtered pseudo-labels
    filtered_scene_tuples = create_scene_tuples(filtered_pl_input_path, filtered_pl_metrics_save_path)
    with Pool(processes=max(1, cpu_count() - 1), maxtasksperchild=1) as pool:
        filtered_results = pool.map(process_scene, filtered_scene_tuples)

    # Summaries for normal pseudo-labels
    total_frames_normal = 0
    scenes_processed_normal = 0
    for (scene_name, success, frame_count) in normal_results:
        if success:
            master_logger.info(f"[NORMAL] Scene '{scene_name}' completed successfully with {frame_count} frames.")
            scenes_processed_normal += 1
            total_frames_normal += frame_count
        else:
            master_logger.error(f"[NORMAL] Scene '{scene_name}' encountered an error. Processed {frame_count} frames.")

    # Summaries for filtered pseudo-labels
    total_frames_filtered = 0
    scenes_processed_filtered = 0
    for (scene_name, success, frame_count) in filtered_results:
        if success:
            master_logger.info(f"[FILTERED] Scene '{scene_name}' completed successfully with {frame_count} frames.")
            scenes_processed_filtered += 1
            total_frames_filtered += frame_count
        else:
            master_logger.error(f"[FILTERED] Scene '{scene_name}' encountered an error. Processed {frame_count} frames.")

    # Compute final aggregated metrics
    mAP_normal, precision_normal, recall_normal = calculate_total_metrics(normal_pl_metrics_save_path)
    mAP_filtered, precision_filtered, recall_filtered = calculate_total_metrics(filtered_pl_metrics_save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Logging overall results
    master_logger.info("=========== Summary of Results ===========")
    master_logger.info(
        f"Normal Scenes Processed: {scenes_processed_normal}/{len(normal_results)}, "
        f"Total Frames: {total_frames_normal}, "
        f"mAP: {mAP_normal:.3f}, precision: {precision_normal:.3f}, recall: {recall_normal:.3f}"
    )
    master_logger.info(
        f"Filtered Scenes Processed: {scenes_processed_filtered}/{len(filtered_results)}, "
        f"Total Frames: {total_frames_filtered}, "
        f"mAP: {mAP_filtered:.3f}, precision: {precision_filtered:.3f}, recall: {recall_filtered:.3f}"
    )
    master_logger.info(f"Total time taken: {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    main()
