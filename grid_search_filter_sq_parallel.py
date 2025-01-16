import os
import pandas as pd
from config import CONFIG
from stats_filter2 import apply_large_sq_filter
import numpy as np
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


def process_frame(scene_path, frame, aspect_ratio, area, ar_dir_name, area_dir_name, base_save_path, task_logger):
    frame_id = frame.split(".")[0]
    frame_path = os.path.join(scene_path, frame)
    try:
        task_logger.info(f"Processing frame: {frame_path}")
        df = pd.read_feather(frame_path)

        df['aspect_ratio'] = df['box_width'] / df['box_length']
        df['area'] = df['box_width'] * df['box_length']
        large_squares_df, rest_boxes_df = apply_large_sq_filter(df, 'aspect_ratio', 'area', aspect_ratio, area)

        save_path = os.path.join(base_save_path, ar_dir_name, area_dir_name, os.path.basename(scene_path), frame_id)
        os.makedirs(save_path, exist_ok=True)

        large_squares_df.to_feather(os.path.join(save_path, "large_squares.feather"))
        rest_boxes_df.to_feather(os.path.join(save_path, "rest_boxes.feather"))
        task_logger.info(f"Frame processed and saved to: {save_path}")
    except Exception as e:
        task_logger.error(f"Error processing frame: {frame_path}. Error: {str(e)}")


def process_scene(input_path, aspect_ratio, area, ar_dir_name, area_dir_name, base_save_path, task_logger):
    try:
        for scene in os.listdir(input_path):
            scene_path = os.path.join(input_path, scene)
            task_logger.info(f"Processing scene: {scene_path}")
            for frame in os.listdir(scene_path):
                process_frame(scene_path, frame, aspect_ratio, area, ar_dir_name, area_dir_name, base_save_path, task_logger)
    except Exception as e:
        task_logger.error(f"Error processing scene: {scene_path}. Error: {str(e)}")


def grid_search_square_filter(input_path, ar_range, area_range, base_save_path, config, main_logger):
    aspect_ratio_dir = config['GRID_SEARCH_SQUARE']['AR_THRESHOLD_BASE_DIR_NAME']
    area_dir = config['GRID_SEARCH_SQUARE']['AREA_THRESHOLD_BASE_DIR_NAME']

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("logs", script_name)
    os.makedirs(log_dir, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        tasks = []
        for aspect_ratio in ar_range:
            ar_dir_name = f"{aspect_ratio_dir}_{round(aspect_ratio, 1)}"  # Explicitly round for clean folder names
            for area in area_range:
                area_dir_name = f"{area_dir}_{round(area, 1)}"  # Explicitly round for clean folder names
                task_log_file = os.path.join(log_dir, f"task_ar_{round(aspect_ratio, 1)}_area_{round(area, 1)}.log")
                task_logger = setup_logger(f"Task_AR_{round(aspect_ratio, 1)}_Area_{round(area, 1)}", task_log_file)
                task_logger.info(f"Starting task for AR: {aspect_ratio}, Area: {area}")
                task = executor.submit(
                    process_scene, input_path, aspect_ratio, area, ar_dir_name, area_dir_name, base_save_path, task_logger
                )
                tasks.append(task)

        # Wait for all tasks to complete
        for task in tasks:
            try:
                task.result()
            except Exception as e:
                main_logger.error(f"Task failed with error: {str(e)}")


def main():
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    log_dir = os.path.join("logs", script_name)
    os.makedirs(log_dir, exist_ok=True)

    main_log_file = os.path.join(log_dir, "main.log")
    main_logger = setup_logger("Main", main_log_file)

    main_logger.info("Starting main process...")
    try:
        home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
        if CONFIG['ROI']:
            ps_base_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['ROI'])
            grid_sq_save_path = os.path.join(home, *CONFIG['GRID_SEARCH_SQUARE']['PATH']['ROI'])
        else:
            ps_base_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['FULL_RANGE'])
            grid_sq_save_path = os.path.join(home, *CONFIG['GRID_SEARCH_SQUARE']['PATH']['FULL_RANGE'])

        aspect_ratio_config = CONFIG['GRID_SEARCH_SQUARE']['ASPECT_RATIO']
        area_config = CONFIG['GRID_SEARCH_SQUARE']['AREA']
        sq_ar_range = np.arange(*aspect_ratio_config['RANGE'], aspect_ratio_config['STEP'])
        sq_area_range = np.arange(*area_config['RANGE'], area_config['STEP'])

        grid_search_square_filter(ps_base_path, sq_ar_range, sq_area_range, grid_sq_save_path, CONFIG, main_logger)
        main_logger.info("Main process completed successfully.")
    except Exception as e:
        main_logger.error(f"Main process failed with error: {str(e)}")


if __name__ == '__main__':
    main()
