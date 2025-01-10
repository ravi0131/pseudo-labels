import logging
import os
import time
from pathlib import Path
import multiprocessing as mp
from functools import partial

import pandas as pd

import av2.datasets.sensor as sensor
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import av2.structures.sweep as sweep

import ge_av2.ground_estimator as ground_exorciser
import ge_av2.utilities as utils

from config import CONFIG


def get_main_logger(script_name: str) -> logging.Logger:
    """
    Creates a master logger that logs high-level information (success/failure of scenes, total time).
    Writes to {script_name}_logs/master.log.
    """
    logs_dir = f"{script_name}_logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "master.log")

    logger = logging.getLogger("main_logger")
    logger.setLevel(logging.INFO)

    # Avoid re-adding handlers on subsequent calls
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_scene_logger(scene_id: str, logs_dir: str) -> logging.Logger:
    """
    Creates a dedicated logger for each scene, storing logs in logs_dir/{scene_id}.log.
    """
    logger_name = f"scene_logger_{scene_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Ensure we don't add duplicate handlers if logger is reused
    if not logger.handlers:
        log_path = os.path.join(logs_dir, f"{scene_id}.log")
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def process_frame(
    frame_path: Path,
    output_dir: str,
    scene_id: str,
    dataset_path: Path,
    scene_logger: logging.Logger
):
    """
    Process a single LIDAR frame to remove ground points and save non-ground points to a .feather file.
    """
    try:
        scene_logger.info(f"Processing frame: {frame_path.stem}")
        lidar_frame = sweep.Sweep.from_feather(frame_path)

        # If config ROI is active, filter points accordingly
        if CONFIG['ROI']:
            points_roi = utils.filter_points_in_ROI(lidar_frame.xyz, **CONFIG['GE_RANGE'])
        else:
            points_roi = lidar_frame.xyz

        # Remove ground
        _, non_ground, _ = ground_exorciser.remove_ground(lidar_frame.xyz, points_roi, percentile=30)

        # Save non-ground points
        non_ground_path = os.path.join(output_dir, scene_id, f"{frame_path.stem}.feather")
        non_ground_df = pd.DataFrame(non_ground, columns=['x', 'y', 'z'])
        non_ground_df.to_feather(non_ground_path)

        scene_logger.info(f"Processed frame {frame_path.stem} -> {len(non_ground_df)} non-ground points.")
        return frame_path.stem

    except Exception as e:
        scene_logger.error(f"Error processing frame {frame_path.stem}: {str(e)}")
        return None


def process_scene(scene_id: str, dataset_path: Path, output_dir: str, logs_dir: str, script_name: str):
    """
    Process all frames for a given scene. Returns a list of successfully processed frames or an empty list on error.
    """
    scene_logger = get_scene_logger(scene_id, logs_dir)
    scene_logger.info(f"Starting scene: {scene_id}")

    try:
        # Prepare data loader
        dataset = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)
        scene_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        # Get lidar frames
        lidar_frames = dataset.get_ordered_log_lidar_fpaths(scene_id)

        # Prepare partial function to include scene_logger
        process_frame_partial = partial(
            process_frame,
            output_dir=output_dir,
            scene_id=scene_id,
            dataset_path=dataset_path,
            scene_logger=scene_logger
        )

        with mp.Pool() as pool:
            results = pool.map(process_frame_partial, lidar_frames)

        # Filter out any frames that returned None (error)
        valid_results = [res for res in results if res is not None]

        scene_logger.info(f"Completed scene {scene_id}: processed {len(valid_results)} frames.")
        return valid_results

    except Exception as e:
        scene_logger.error(f"Failed to process scene {scene_id}: {str(e)}")
        return []


def main():
    # Derive the script name or default to e.g. "av2_ground_estimation"
    script_name = os.path.splitext(os.path.basename(__file__))[0] \
        if "__file__" in globals() else "av2_ground_estimation"

    # Create the master logger
    main_logger = get_main_logger(script_name)

    # Record the start time
    start_time = time.time()

    # Setup paths
    dataset_path = Path(os.path.join(os.path.expanduser("~"), "buni", "dataset", "av2", "train"))
    output_dir = os.path.join(os.path.expanduser("~"), 'buni', 'output-data', 'av2', CONFIG['GE_EXPORT_DIR'])
    os.makedirs(output_dir, exist_ok=True)

    # Directory for all logs (master log + scene logs)
    logs_dir = f"{script_name}_logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize dataset and get scene IDs
    dataset = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)
    scene_ids = dataset.get_log_ids()

    scene_count = 10  # CONFIG PARAMETER: Number of scenes to process
    scene_ids = scene_ids[:scene_count]

    main_logger.info(f"Processing {len(scene_ids)} scenes.")

    total_frames_processed = 0

    # Process each scene
    for scene_id in scene_ids:
        main_logger.info(f"Processing scene: {scene_id}")
        processed_frames = process_scene(scene_id, dataset_path, output_dir, logs_dir, script_name)
        main_logger.info(f"Scene '{scene_id}' completed. Processed {len(processed_frames)} frames.")
        total_frames_processed += len(processed_frames)

    # Record the end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    main_logger.info("All scene processing complete.")
    main_logger.info(f"Total frames processed: {total_frames_processed}")
    main_logger.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
