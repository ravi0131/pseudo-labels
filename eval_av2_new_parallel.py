import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from config import CONFIG
from prototype_utils import (
    filter_cuboids_by_roi,
    extract_face_corners,
    bboxes_df_to_numpy_corners,
    filter_gt_labels_by_category,
)
from eval_metrics import compute_matches, compute_ap2


# ------------------------------------------------------------------------------
# 1) Utility: Setup a file-based logger for each scene
# ------------------------------------------------------------------------------
def setup_logger(scene_id: str, log_file: str) -> logging.Logger:
    """
    Create and return a logger that writes to `log_file`.
    Each scene can have its own log file.
    """
    logger = logging.getLogger(scene_id)
    logger.setLevel(logging.INFO)
    # Avoid adding multiple FileHandlers if logger is reused
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] (%(name)s): %(message)s"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.propagate = False  # Don't propagate to root logger
    return logger


# ------------------------------------------------------------------------------
# 2) Frame-level calculations (UNCHANGED)
#    calculate_metrics_frame_normal / calculate_metrics_frame_filtered
# ------------------------------------------------------------------------------
def calculate_metrics_frame_normal(
    input_pl_frame_path: str,
    frame_id: str,
    scene_id: str,
    scene_save_path: str,
    av2: AV2SensorDataLoader,
    config: Dict,
    iou_threshold: float
):
    try:
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)

        if config["ROI"]:
            filtered_cuboids = filter_cuboids_by_roi(relevant_cuboids.vertices_m, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            gt_corners = extract_face_corners(relevant_cuboids.vertices_m)

        # 2) Load pseudo-labels
        pseudo_labels_df = pd.read_feather(input_pl_frame_path)
        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)

        if iou_threshold not in [0.3, 0.5]:
            raise ValueError(
                f"iou_threshold should be either 0.3 or 0.5. Got {iou_threshold}."
            )

        
        _, pseudo_label_matches, _ = compute_matches(
            gt_corners, pseudo_labels_corners, iou_threshold=iou_threshold
        )

        gt_length = len(gt_corners)
        num_of_predictions = len(pseudo_labels_corners)

        (
            mAP,
            precisions,
            recalls,
            precision,
            recall,
        ) = compute_ap2(pseudo_label_matches, gt_length, num_of_predictions)

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
            raise ValueError(
                f"Path {scene_save_path} does not exist. Please create it first."
            )

        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        results_df.to_feather(os.path.join(frame_save_path, f"iou_{iou_threshold}_.feather"))

    except Exception as e:
        # If "filtered_cuboids" was not defined, avoid referencing it
        print(
            f"NORMAL FRAME: Error processing frame {frame_id} in scene {scene_id}: {e}"
        )


def calculate_metrics_frame_filtered(
    input_pl_frame_path: str,
    frame_id: str,
    scene_id: str,
    scene_save_path: str,
    av2: AV2SensorDataLoader,
    config: Dict,
    iou_threshold: float
):
    try:
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)

        if config["ROI"]:
            filtered_cuboids = filter_cuboids_by_roi(relevant_cuboids.vertices_m, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            gt_corners = extract_face_corners(relevant_cuboids.vertices_m)

        # Load pre-filtered pseudo-labels
        if iou_threshold == 0.3:
            pseudo_labels_df = pd.read_feather(
                os.path.join(input_pl_frame_path, "iou_0.3_.feather")
            )
        elif iou_threshold == 0.5:
            pseudo_labels_df = pd.read_feather(
                os.path.join(input_pl_frame_path, "iou_0.5_.feather")
            )
        else:
            raise ValueError("iou_threshold should be 0.3 or 0.5")

        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)

        _, pseudo_label_matches, _ = compute_matches(
            gt_corners, pseudo_labels_corners, iou_threshold=float(iou_threshold)
        )

        gt_length = len(gt_corners)
        num_of_predictions = len(pseudo_labels_corners)

        (
            mAP,
            precisions,
            recalls,
            precision,
            recall,
        ) = compute_ap2(pseudo_label_matches, gt_length, num_of_predictions)

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
            raise ValueError(
                f"Path {scene_save_path} does not exist. Please create it first."
            )

        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        results_df.to_feather(os.path.join(frame_save_path, f"iou_{iou_threshold}_.feather"))

    except Exception as e:
        print(
            f"FILTERED FRAME: Error processing frame {frame_id} in scene {scene_id}: {e}"
        )


# ------------------------------------------------------------------------------
# 3) Scene-level logic (UNCHANGED)
# ------------------------------------------------------------------------------
def calculate_metrics_scene_normal(
    input_pl_path: str,
    scene_id: str,
    base_save_path: str,
    av2: AV2SensorDataLoader,
    config: Dict,
    iou_threshold: float,
    logger: logging.Logger = None
):
    scene_save_path = os.path.join(base_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)

    frames = os.listdir(input_pl_path)
    for frame in frames:
        frame_id = os.path.splitext(frame)[0]
        frame_path = os.path.join(input_pl_path, frame)
        if logger:
            logger.info(f"[NORMAL] Processing frame {frame_id} in scene {scene_id}")
        calculate_metrics_frame_normal(
            frame_path,
            frame_id,
            scene_id,
            scene_save_path,
            av2,
            config,
            iou_threshold
        )


def calculate_metrics_scene_filtered(
    input_pl_path: str,
    scene_id: str,
    base_save_path: str,
    av2: AV2SensorDataLoader,
    config: Dict,
    iou_threshold: float,
    logger: logging.Logger = None
):
    scene_save_path = os.path.join(base_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)

    frames = os.listdir(input_pl_path)
    for frame in frames:
        frame_id = os.path.splitext(frame)[0]
        frame_path = os.path.join(input_pl_path, frame)
        if logger:
            logger.info(f"[FILTERED] Processing frame {frame_id} in scene {scene_id}")
        calculate_metrics_frame_filtered(
            frame_path,
            frame_id,
            scene_id,
            scene_save_path,
            av2,
            config,
            iou_threshold
        )


# ------------------------------------------------------------------------------
# 4) Parallel Wrappers at the Scene Level (with Logging)
# ------------------------------------------------------------------------------
def parallel_calculate_all_metrics_framewise_normal(
    input_pseudo_labels_path: str,
    av2: AV2SensorDataLoader,
    base_save_path: str,
    config: Dict,
    iou_threshold: float,
    logs_dir: str
):
    """
    Parallel version of the normal pseudo-label metric calculation.
    Spawns one process per scene, each writing logs to logs_dir/scene_<ID>_normal.log
    """
    os.makedirs(base_save_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures_map = {}
    with ProcessPoolExecutor() as executor:
        for scene_id in os.listdir(input_pseudo_labels_path):
            scene_input_path = os.path.join(input_pseudo_labels_path, scene_id)
            if not os.path.isdir(scene_input_path):
                continue

            # Each scene logs to logs_dir/scene_{scene_id}_normal.log
            log_file_path = os.path.join(logs_dir, f"scene_{scene_id}_normal.log")

            future = executor.submit(
                process_scene_normal_with_logger,
                scene_input_path,
                scene_id,
                base_save_path,
                av2,
                config,
                iou_threshold,
                log_file_path
            )
            futures_map[future] = scene_id

    for future in as_completed(futures_map):
        sid = futures_map[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error in normal scene {sid}: {e}")


def parallel_calculate_all_metrics_framewise_filtered(
    input_pseudo_labels_path: str,
    av2: AV2SensorDataLoader,
    base_save_path: str,
    config: Dict,
    iou_threshold: float,
    logs_dir: str
):
    """
    Parallel version of the filtered pseudo-label metric calculation.
    Spawns one process per scene, each writing logs to logs_dir/scene_<ID>_filtered.log
    """
    os.makedirs(base_save_path, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    futures_map = {}
    with ProcessPoolExecutor() as executor:
        for scene_id in os.listdir(input_pseudo_labels_path):
            scene_input_path = os.path.join(input_pseudo_labels_path, scene_id)
            if not os.path.isdir(scene_input_path):
                continue

            # Each scene logs to logs_dir/scene_{scene_id}_filtered.log
            log_file_path = os.path.join(logs_dir, f"scene_{scene_id}_filtered.log")

            future = executor.submit(
                process_scene_filtered_with_logger,
                scene_input_path,
                scene_id,
                base_save_path,
                av2,
                config,
                iou_threshold,
                log_file_path
            )
            futures_map[future] = scene_id

    for future in as_completed(futures_map):
        sid = futures_map[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error in filtered scene {sid}: {e}")


# ------------------------------------------------------------------------------
# 5) Helper functions that set up a logger, then call scene-level code
#    We separate these so each process can create & use its own logger.
# ------------------------------------------------------------------------------
def process_scene_normal_with_logger(
    scene_input_path: str,
    scene_id: str,
    base_save_path: str,
    av2: AV2SensorDataLoader,
    config: Dict,
    iou_threshold: float,
    log_file_path: str
):
    logger = setup_logger(f"normal_{scene_id}", log_file_path)
    logger.info(f"Processing Normal Scene {scene_id} with IoU={iou_threshold}")
    try:
        calculate_metrics_scene_normal(
            scene_input_path,
            scene_id,
            base_save_path,
            av2,
            config,
            iou_threshold,
            logger
        )
        logger.info(f"Completed Normal Scene {scene_id} successfully.")
    except Exception as e:
        logger.error(f"Scene {scene_id} failed: {e}", exc_info=True)
        raise


def process_scene_filtered_with_logger(
    scene_input_path: str,
    scene_id: str,
    base_save_path: str,
    av2: AV2SensorDataLoader,
    config: Dict,
    iou_threshold: float,
    log_file_path: str
):
    logger = setup_logger(f"filtered_{scene_id}", log_file_path)
    logger.info(f"Processing Filtered Scene {scene_id} with IoU={iou_threshold}")
    try:
        calculate_metrics_scene_filtered(
            scene_input_path,
            scene_id,
            base_save_path,
            av2,
            config,
            iou_threshold,
            logger
        )
        logger.info(f"Completed Filtered Scene {scene_id} successfully.")
    except Exception as e:
        logger.error(f"Scene {scene_id} failed: {e}", exc_info=True)
        raise


# ------------------------------------------------------------------------------
# 6) Summation of metrics (UNCHANGED)
# ------------------------------------------------------------------------------
def calculate_total_metrics_normal(base_save_path: str, iou_threshold: float):
    mAPs = []
    precisions = []
    recalls = []

    if str(iou_threshold) == "0.3":
        df_filename = "iou_0.3_.feather"
    elif str(iou_threshold) == "0.5":
        df_filename = "iou_0.5_.feather"
    else:
        raise ValueError(f"iou_threshold must be 0.3 or 0.5, got {iou_threshold}")

    for scene_id in os.listdir(base_save_path):
        scene_path = os.path.join(base_save_path, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for frame in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame)
            if not os.path.isdir(frame_path):
                continue
            results_file = os.path.join(frame_path, df_filename)
            if not os.path.isfile(results_file):
                continue

            df = pd.read_feather(results_file)
            mAPs.append(df['mAP'][0])
            precisions.append(df['precision'][0])
            recalls.append(df['recall'][0])

    if len(mAPs) == 0:
        return 0.0, 0.0, 0.0
    mAP = sum(mAPs) / len(mAPs)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return mAP, precision, recall


# ------------------------------------------------------------------------------
# 7) Main
# ------------------------------------------------------------------------------
def main(iou_mode: float, av2: AV2SensorDataLoader):
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])


    # 1) Determine input & output paths
    
    # 1) Determine input & output paths
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

    # We define two logs directories for normal/filtered so logs don’t collide
    normal_logs_dir = os.path.join("logs", f"normal_{iou_mode}")
    filtered_logs_dir = os.path.join("logs", f"filtered_{iou_mode}")

    # 1) Parallel scene-level metric calculations
    parallel_calculate_all_metrics_framewise_normal(
        normal_pl_input_path,
        av2,
        normal_pl_metrics_save_path,
        CONFIG,
        iou_threshold=iou_mode,
        logs_dir=normal_logs_dir
    )
    parallel_calculate_all_metrics_framewise_filtered(
        filtered_pl_input_path,
        av2,
        filtered_pl_metrics_save_path,
        CONFIG,
        iou_threshold=iou_mode,
        logs_dir=filtered_logs_dir
    )

    # 2) Summarize total metrics
    mAP_normal, precision_normal, recall_normal = calculate_total_metrics_normal(
        normal_pl_metrics_save_path,
        iou_threshold=iou_mode
    )
    mAP_filtered, precision_filtered, recall_filtered = calculate_total_metrics_normal(
        filtered_pl_metrics_save_path,
        iou_threshold=iou_mode
    )

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
    print(
        f"Normal Pseudo-labels: mAP = {normal_metrics_03['mAP']:.4f}, "
        f"Precision = {normal_metrics_03['precision']:.4f}, "
        f"Recall = {normal_metrics_03['recall']:.4f}"
    )
    print(
        f"Filtered Pseudo-labels: mAP = {filtered_metrics_03['mAP']:.4f}, "
        f"Precision = {filtered_metrics_03['precision']:.4f}, "
        f"Recall = {filtered_metrics_03['recall']:.4f}"
    )
    
    print("\n--- Results for IoU = 0.5 ---\n")
    print(
        f"Normal Pseudo-labels: mAP = {normal_metrics_05['mAP']:.4f}, "
        f"Precision = {normal_metrics_05['precision']:.4f}, "
        f"Recall = {normal_metrics_05['recall']:.4f}"
    )
    print(
        f"Filtered Pseudo-labels: mAP = {filtered_metrics_05['mAP']:.4f}, "
        f"Precision = {filtered_metrics_05['precision']:.4f}, "
        f"Recall = {filtered_metrics_05['recall']:.4f}"
    )
