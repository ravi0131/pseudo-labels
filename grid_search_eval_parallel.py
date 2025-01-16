import os
import pandas as pd
from config import CONFIG
from typing import Dict
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from prototype_utils import filter_cuboids_by_roi, extract_face_corners, filter_gt_labels_by_category, bboxes_df_to_numpy_corners
from eval_metrics import compute_matches, compute_ap2
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor


# Set up logging directories
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False  # Avoid duplicate logs in the root logger
    return logger


def grid_search_compute_metrics_frame(input_frame_path: str, scene_id: str, frame_id: str, frame_save_path: str,
                                      av2: AV2SensorDataLoader, config: Dict, task_logger):
    try:
        task_logger.info(f"Processing frame {frame_id} in scene {scene_id}")
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
        filename_map = {
            ('rect_filter', 'positive'): "other_boxes.feather",
            ('rect_filter', 'negative'): "narrow_boxes.feather",
            ('square_filter', 'positive'): "rest_boxes.feather",
            ('square_filter', 'negative'): "large_squares.feather",
        }
        filename = filename_map.get((mode['FILTER'], mode['LABEL_TYPE']))
        if not filename:
            raise ValueError(f"Invalid mode: {mode}")

        pseudo_labels_df = pd.read_feather(os.path.join(input_frame_path, filename))
        pseudo_labels_corners = bboxes_df_to_numpy_corners(pseudo_labels_df)

        _, pseudo_label_matches1, _ = compute_matches(gt_corners, pseudo_labels_corners, iou_threshold=0.3)
        _, pseudo_label_matches2, _ = compute_matches(gt_corners, pseudo_labels_corners, iou_threshold=0.5)

        gt_length = len(gt_corners)
        num_of_predictions = len(pseudo_labels_corners)

        mAP1, precisions1, recalls1, precision1, recall1 = compute_ap2(pseudo_label_matches1, gt_length, num_of_predictions)
        mAP2, precisions2, recalls2, precision2, recall2 = compute_ap2(pseudo_label_matches2, gt_length, num_of_predictions)

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

        os.makedirs(frame_save_path, exist_ok=True)
        results_df_1.to_feather(os.path.join(frame_save_path, f"iou_0.3_.feather"))
        results_df_2.to_feather(os.path.join(frame_save_path, f"iou_0.5_.feather"))

        task_logger.info(f"Metrics computed and saved for frame {frame_id} in scene {scene_id}")
    except Exception as e:
        task_logger.error(f"Error processing frame {frame_id} in scene {scene_id}: {e}")


def grid_search_compute_metrics_scene(input_scene_path, scene_id, output_scene_path, av2: AV2SensorDataLoader, config: Dict, task_logger):
    for frame_id in os.listdir(input_scene_path):
        input_frame_path = os.path.join(input_scene_path, frame_id)
        output_frame_path = os.path.join(output_scene_path, frame_id)
        os.makedirs(output_frame_path, exist_ok=True)
        grid_search_compute_metrics_frame(input_frame_path, scene_id, frame_id, output_frame_path, av2, config, task_logger)


def grid_search_compute_metrics_for_a_combination(base_scene_path: str, base_output_path: str, av2: AV2SensorDataLoader, config: Dict, task_logger):
    for scene_id in os.listdir(base_scene_path):
        input_scene_path = os.path.join(base_scene_path, scene_id)
        output_scene_path = os.path.join(base_output_path, scene_id)
        os.makedirs(output_scene_path, exist_ok=True)
        grid_search_compute_metrics_scene(input_scene_path, scene_id, output_scene_path, av2, config, task_logger)


def grid_search_compute_metrics_for_filter(input_base_path: str, base_output_path: str, av2: AV2SensorDataLoader, config: Dict, main_logger):
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("logs", script_name)
    os.makedirs(log_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        tasks = []
        for ar_thres_dir in os.listdir(input_base_path):
            ar_threshold_value = ar_thres_dir.split("_")[-1]
            ar_threshold_value_path = os.path.join(input_base_path, ar_thres_dir)
            for area_thres_dir in os.listdir(ar_threshold_value_path):
                area_threshold_value = area_thres_dir.split("_")[-1]
                area_threshold_value_path = os.path.join(ar_threshold_value_path, area_thres_dir)
                combination_output_path = os.path.join(base_output_path, ar_thres_dir, area_thres_dir)

                task_log_file = os.path.join(log_dir, f"task_ar_{ar_threshold_value}_area_{area_threshold_value}.log")
                task_logger = setup_logger(f"Task_AR_{ar_threshold_value}_Area_{area_threshold_value}", task_log_file)

                task_logger.info(f"Starting task for AR: {ar_threshold_value}, Area: {area_threshold_value}")
                task = executor.submit(
                    grid_search_compute_metrics_for_a_combination,
                    area_threshold_value_path,
                    combination_output_path,
                    av2,
                    config,
                    task_logger
                )
                tasks.append(task)

        # Wait for all tasks to complete
        for task in tasks:
            try:
                task.result()
            except Exception as e:
                main_logger.error(f"Task failed with error: {e}")


def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("logs", script_name)
    os.makedirs(log_dir, exist_ok=True)

    main_log_file = os.path.join(log_dir, "main.log")
    main_logger = setup_logger("Main", main_log_file)

    main_logger.info("Starting grid search metrics computation...")
    try:
        home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
        filter_type = CONFIG['GRID_SEARCH_METRICS_MODE']['FILTER']

        if CONFIG['ROI']:
            if filter_type == 'rect_filter':
                filter_base_path = os.path.join(home, *CONFIG["GRID_SEARCH_RECT"]['PATH']['ROI'])
                grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['ROI']['RECT_FILTER_PATH'])
            else:
                filter_base_path = os.path.join(home, *CONFIG["GRID_SEARCH_SQUARE"]['PATH']['ROI'])
                grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['ROI']['SQUARE_FILTER_PATH'])
        else:
            if filter_type == 'rect_filter':
                filter_base_path = os.path.join(home, *CONFIG["GRID_SEARCH_RECT"]['PATH']['FULL_RANGE'])
                grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['FULL_RANGE']['RECT_FILTER_PATH'])
            else:
                filter_base_path = os.path.join(home, *CONFIG["GRID_SEARCH_SQUARE"]['PATH']['FULL_RANGE'])
                grid_search_metrics_output_path = os.path.join(home, *CONFIG["GRID_SEARCH_METRICS_PATH"]['FULL_RANGE']['SQUARE_FILTER_PATH'])

        av2_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))
        av2 = AV2SensorDataLoader(data_dir=av2_path, labels_dir=av2_path)

        grid_search_compute_metrics_for_filter(filter_base_path, grid_search_metrics_output_path, av2, CONFIG, main_logger)
        main_logger.info("Grid search metrics computation completed successfully.")
    except Exception as e:
        main_logger.error(f"Main process failed with error: {e}")


if __name__ == '__main__':
    main()
