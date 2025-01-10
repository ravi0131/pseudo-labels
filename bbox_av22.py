import logging
import os
import pandas as pd
from bboxer_av2 import bboxer
from bboxer_av2 import utils as box_utils
import time
from multiprocessing import Pool, cpu_count
from functools import partial
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
    
    # Avoid adding multiple handlers if logger is requested more than once
    if not logger.handlers:
        log_path = os.path.join(logs_dir, f"{scene_name}.log")
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_bboxes_to_feather(rects_modest, savepath, frame_id, scene_logger):
    data = {
        "box_center_x": [rect['box_center'][0] for rect in rects_modest],
        "box_center_y": [rect['box_center'][1] for rect in rects_modest],
        "box_length": [rect['box_length'] for rect in rects_modest],
        "box_width": [rect['box_width'] for rect in rects_modest],
        "ry": [rect['ry'] for rect in rects_modest]
    }
    
    df = pd.DataFrame(data)
    scene_logger.info(f"Dataframe size before saving (frame={frame_id}): {df.shape}")
    save_path = os.path.join(savepath, frame_id)
    df.to_feather(save_path)

def process_frame(frame: str, scene_path: str, output_dir: str, scene_logger: logging.Logger):
    """
    Process a single frame to estimate bounding boxes.
    """
    try:
        frame_path = os.path.join(scene_path, frame)  # test_ge_script/scene_id/frame_id.feather
        save_dir = os.path.join(output_dir, os.path.basename(scene_path))  # test_bbox_script/scene_id
        os.makedirs(save_dir, exist_ok=True)
        
        non_ground_points = pd.read_feather(frame_path)
        
        bbox_estimator = bboxer.Bboxer()
        bbox_estimator.cluster(non_ground_points.to_numpy()[:, :2])
        rects = bbox_estimator.estimate_bboxes_from_clusters_modest(
            bbox_estimator.clustered_points,
            bbox_estimator.clustered_labels,
            'closeness_to_edge'
        )
        
        scene_logger.info(f"Processing frame {frame}: Found {len(rects)} bounding boxes")
        
        if os.access(save_dir, os.W_OK):
            save_bboxes_to_feather(rects, save_dir, frame, scene_logger)
        else:
            scene_logger.error(f"Cannot write to directory {save_dir}")
        
        return frame
    except Exception as e:
        scene_logger.error(f"Error processing frame {frame}: {str(e)}")
        return None

def process_scene(scene_tuple):
    """
    Process a single scene sequentially (in a single worker).
    """
    scene_path, output_dir, logs_dir, script_name = scene_tuple
    
    scene_name = os.path.basename(scene_path)
    scene_logger = get_scene_logger(scene_name, logs_dir)
    
    frames_in_scene = []
    
    try:
        scene_logger.info(f"Starting scene: {scene_name}")
        
        # Process frames
        for frame in os.listdir(scene_path):
            result = process_frame(frame, scene_path, output_dir, scene_logger)
            if result is not None:
                frames_in_scene.append(result)
        
        scene_logger.info(f"Completed scene {scene_name}: {len(frames_in_scene)} frames processed")
        
        return (scene_name, True, len(frames_in_scene))
    
    except Exception as e:
        scene_logger.error(f"Error processing scene {scene_name}: {str(e)}")
        return (scene_name, False, len(frames_in_scene))

def main():
    # Derive your script name. You could also just do script_name = "bbox_estimation"
    script_name = os.path.splitext(os.path.basename(__file__))[0]  \
        if "__file__" in globals() else "bbox_estimation"
    
    # Directory that will hold all logs
    logs_dir = f"{script_name}_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Master logger logs high-level info only
    master_logger = get_main_logger(script_name)
    
    home = os.path.expanduser("~")
    ge_data_dir = os.path.join(home, 'buni', 'output-data', 'av2', 'test_ge_script')
    output_dir = os.path.join(home, 'buni', 'output-data', 'av2', 'test_bbox_script')
    
    start_time = time.time()
    
    # Create list of scene tuples
    scene_tuples = [
        (os.path.join(ge_data_dir, scene), output_dir, logs_dir, script_name)
        for scene in os.listdir(ge_data_dir)
    ]
    
    scene_count = CONFIG['SCENE_COUNT']  # CONFIG PARAMETER: Number of scenes to process
    scene_tuples = scene_tuples[:scene_count]
    
    # Parallel processing
    with Pool(processes=max(1, cpu_count() - 1), maxtasksperchild=1) as pool:
        results = pool.map(process_scene, scene_tuples)
    
    # Summarize results in the master logger
    total_frames = 0
    scenes_processed = 0
    for (scene_name, success, frame_count) in results:
        if success:
            master_logger.info(f"Scene '{scene_name}' completed successfully with {frame_count} frames.")
            scenes_processed += 1
            total_frames += frame_count
        else:
            master_logger.error(f"Scene '{scene_name}' encountered an error. Processed {frame_count} frames.")
    
    end_time = time.time()
    master_logger.info(f"Processing complete.")
    master_logger.info(f"Number of scenes processed successfully: {scenes_processed}/{len(results)}")
    master_logger.info(f"Total frames processed: {total_frames}")
    master_logger.info(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
