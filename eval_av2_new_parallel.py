from prototype_utils import (
    filter_cuboids_by_roi,
    extract_face_corners,
    bboxes_df_to_numpy_corners,
    filter_gt_labels_by_category,
)
from eval_metrics import compute_matches, compute_ap2
from typing import Dict
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import os
import pandas as pd
from config import CONFIG

# For parallelism
from concurrent.futures import ProcessPoolExecutor, as_completed


# ------------------------------------------------------------------------------
# 1) Per-frame metrics: (Normal / Filtered) -- UNCHANGED
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
    """
    Calculate the metrics for a single frame (normal pseudo-labels).
    Saves result to scene_save_path/frame_id/iou_{iou_threshold}_.feather
    """
    try:
        # 1) Grab ground truth
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

        # 3) Compute matches
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

        # 4) Write out
        if not os.path.exists(scene_save_path):
            raise ValueError(
                f"Path {scene_save_path} does not exist. "
                "Please create the directory before calling this function"
            )
        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        results_df.to_feather(os.path.join(frame_save_path, f"iou_{iou_threshold}_.feather"))

    except Exception as e:
        print(
            f"NORMAL FRAME: Error processing frame {frame_id} "
            f"in scene {scene_id} : {e}"
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
    """
    Calculate the metrics for a single frame (filtered pseudo-labels).
    Saves result to scene_save_path/frame_id/iou_{iou_threshold}_.feather
    """
    try:
        # 1) Grab ground truth
        cuboids = av2.get_labels_at_lidar_timestamp(scene_id, int(frame_id))
        relevant_cuboids = filter_gt_labels_by_category(cuboids, config)
        if config["ROI"]:
            filtered_cuboids = filter_cuboids_by_roi(relevant_cuboids.vertices_m, config)
            gt_corners = extract_face_corners(filtered_cuboids)
        else:
            gt_corners = extract_face_corners(relevant_cuboids.vertices_m)

        # 2) Load pre-filtered pseudo-labels
        if iou_threshold == 0.3:
            pseudo_labels_df = pd.read_feather(
                os.path.join(input_pl_frame_path, "iou_0.3_.feather")
            )
        elif iou_threshold == 0.5:
            pseudo_labels_df = pd.read_feather(
                os.path.join(input_pl_frame_path, "iou_0.5_.feather")
            )
        else:
            raise ValueError("iou_threshold must be 0.3 or 0.5")

        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)

        # 3) Compute matches
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

        # 4) Write out
        if not os.path.exists(scene_save_path):
            raise ValueError(
                f"Path {scene_save_path} does not exist. "
                "Please create the directory before calling this function"
            )
        frame_save_path = os.path.join(scene_save_path, frame_id)
        os.makedirs(frame_save_path, exist_ok=True)
        results_df.to_feather(os.path.join(frame_save_path, f"iou_{iou_threshold}_.feather"))

    except Exception as e:
        print(
            f"FILTERED FRAME: Error processing frame {frame_id} "
            f"in scene {scene_id} : {e}"
        )


# ------------------------------------------------------------------------------
# 2) Per-scene metrics: (Normal / Filtered) -- UNCHANGED
# ------------------------------------------------------------------------------

def calculate_metrics_scene_normal(
    input_pl_path: str,
    scene_id: str,
    base_save_path: str,
    av2: AV2SensorDataLoader,
    iou_threshold: float
):
    """
    For each frame in 'input_pl_path', call `calculate_metrics_frame_normal`.
    """
    scene_save_path = os.path.join(base_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)
    for frame in os.listdir(input_pl_path):
        frame_path = os.path.join(input_pl_path, frame)
        frame_id = os.path.splitext(frame)[0]
        calculate_metrics_frame_normal(
            frame_path, frame_id, scene_id, scene_save_path, av2, CONFIG, iou_threshold
        )


def calculate_metrics_scene_filtered(
    input_pl_path: str,
    scene_id: str,
    base_save_path: str,
    av2: AV2SensorDataLoader,
    iou_threshold: float
):
    """
    For each frame in 'input_pl_path', call `calculate_metrics_frame_filtered`.
    """
    scene_save_path = os.path.join(base_save_path, scene_id)
    os.makedirs(scene_save_path, exist_ok=True)
    for frame in os.listdir(input_pl_path):
        frame_path = os.path.join(input_pl_path, frame)
        frame_id = os.path.splitext(frame)[0]
        calculate_metrics_frame_filtered(
            frame_path, frame_id, scene_id, scene_save_path, av2, CONFIG, iou_threshold
        )


# ------------------------------------------------------------------------------
# 3) Parallelize scene-level processing for normal & filtered
# ------------------------------------------------------------------------------

def parallel_calculate_all_metrics_framewise_normal(
    input_pseudo_labels_path: str,
    av2: AV2SensorDataLoader,
    base_save_path: str,
    iou_threshold: float
):
    """
    Parallel version of calculate_all_metrics_framewise_normal.
    Spawns one process per scene.
    """
    os.makedirs(base_save_path, exist_ok=True)

    futures = []
    with ProcessPoolExecutor() as executor:
        for scene_id in os.listdir(input_pseudo_labels_path):
            scene_input_path = os.path.join(input_pseudo_labels_path, scene_id)
            if not os.path.isdir(scene_input_path):
                continue

            # For each scene, we submit a job to process it
            future = executor.submit(
                calculate_metrics_scene_normal,
                scene_input_path,
                scene_id,
                base_save_path,
                av2,
                iou_threshold
            )
            futures.append((future, scene_id))

        for future, scene_id in as_completed(dict(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error in normal scene {scene_id}: {e}")


def parallel_calculate_all_metrics_framewise_filtered(
    input_pseudo_labels_path: str,
    av2: AV2SensorDataLoader,
    base_save_path: str,
    iou_threshold: float
):
    """
    Parallel version of calculate_all_metrics_framewise_filtered.
    Spawns one process per scene.
    """
    os.makedirs(base_save_path, exist_ok=True)

    futures = []
    with ProcessPoolExecutor() as executor:
        for scene_id in os.listdir(input_pseudo_labels_path):
            scene_input_path = os.path.join(input_pseudo_labels_path, scene_id)
            if not os.path.isdir(scene_input_path):
                continue

            # For each scene, we submit a job
            future = executor.submit(
                calculate_metrics_scene_filtered,
                scene_input_path,
                scene_id,
                base_save_path,
                av2,
                iou_threshold
            )
            futures.append((future, scene_id))

        for future, scene_id in as_completed(dict(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error in filtered scene {scene_id}: {e}")


# ------------------------------------------------------------------------------
# 4) Summarize total metrics (unchanged)
# ------------------------------------------------------------------------------

def calculate_total_metrics_normal(base_save_path: str, iou_threshold: float):
    """
    Calculate the total metrics for all scenes at the given IOU threshold.
    """
    mAPs = []
    precisions = []
    recalls = []

    # Determine the file to read
    if str(iou_threshold) == "0.3":
        df_filename = "iou_0.3_.feather"
    elif str(iou_threshold) == "0.5":
        df_filename = "iou_0.5_.feather"
    else:
        raise ValueError("iou_threshold must be 0.3 or 0.5")

    for scene_id in os.listdir(base_save_path):
        scene_path = os.path.join(base_save_path, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for frame_id in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame_id)
            if not os.path.isdir(frame_path):
                continue
            results_file = os.path.join(frame_path, df_filename)
            if not os.path.isfile(results_file):
                continue

            results_df = pd.read_feather(results_file)
            mAPs.append(results_df["mAP"][0])
            precisions.append(results_df["precision"][0])
            recalls.append(results_df["recall"][0])

    if len(mAPs) == 0:
        return 0.0, 0.0, 0.0
    mAP = sum(mAPs) / len(mAPs)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    return mAP, precision, recall


# ------------------------------------------------------------------------------
# 5) Main
# ------------------------------------------------------------------------------

def main(iou_mode: float, av2: AV2SensorDataLoader):
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])

    # 1) Determine input & output paths
    if CONFIG['ROI']:
        normal_pl_input_path = os.path.join(home, *CONFIG["BBOX_FILE_PATHS"]['ROI'])
        filtered_pl_input_path = os.path.join(home, *CONFIG["FILTERED_BBOX_FILE_PATHS"]['ROI'])

        normal_pl_metrics_save_path = os.path.join(home, *CONFIG['METRICS_FILE_PATHS']['ROI']['NORMAL'])
        filtered_pl_metrics_save_path = os.path.join(home, *CONFIG['METRICS_FILE_PATHS']['ROI']['FILTERED'])
    else:
        normal_pl_input_path = os.path.join(home, *CONFIG["BBOX_FILE_PATHS"]['FULL_RANGE'])
        filtered_pl_input_path = os.path.join(home, *CONFIG["FILTERED_BBOX_FILE_PATHS"]['FULL_RANGE'])

        normal_pl_metrics_save_path = os.path.join(
            home, *CONFIG['METRICS_FILE_PATHS']['FULL_RANGE']['NORMAL']
        )
        filtered_pl_metrics_save_path = os.path.join(
            home, *CONFIG['METRICS_FILE_PATHS']['FULL_RANGE']['FILTERED']
        )

    # 2) Calculate metrics in parallel for all scenes (Normal + Filtered)
    parallel_calculate_all_metrics_framewise_normal(
        normal_pl_input_path,
        av2,
        normal_pl_metrics_save_path,
        iou_threshold=iou_mode
    )
    parallel_calculate_all_metrics_framewise_filtered(
        filtered_pl_input_path,
        av2,
        filtered_pl_metrics_save_path,
        iou_threshold=iou_mode
    )

    # 3) Aggregate total metrics
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


# ------------------------------------------------------------------------------
# 6) Entry point
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    dataset_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))
    av2 = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)

    normal_metrics_03, filtered_metrics_03 = main(0.3, av2)
    normal_metrics_05, filtered_metrics_05 = main(0.5, av2)

    print("\n--- Results for IoU = 0.3 ---\n")
    print(
        f"Normal Pseudo-labels: mAP = {normal_metrics_03['mAP']}, "
        f"Precision = {normal_metrics_03['precision']}, "
        f"Recall = {normal_metrics_03['recall']}"
    )
    print(
        f"Filtered Pseudo-labels: mAP = {filtered_metrics_03['mAP']}, "
        f"Precision = {filtered_metrics_03['precision']}, "
        f"Recall = {filtered_metrics_03['recall']}"
    )

    print("\n--- Results for IoU = 0.5 ---\n")
    print(
        f"Normal Pseudo-labels: mAP = {normal_metrics_05['mAP']}, "
        f"Precision = {normal_metrics_05['precision']}, "
        f"Recall = {normal_metrics_05['recall']}"
    )
    print(
        f"Filtered Pseudo-labels: mAP = {filtered_metrics_05['mAP']}, "
        f"Precision = {filtered_metrics_05['precision']}, "
        f"Recall = {filtered_metrics_05['recall']}"
    )
