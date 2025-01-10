import logging
import os
import pandas as pd
from bboxer_av2 import bboxer
from bboxer_av2 import utils as box_utils
import time
from multiprocessing import Pool, cpu_count
from functools import partial

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(os.getcwd(), 'bbox_estimation.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def save_bboxes_to_feather(rects_modest, savepath, frame_id, logger):
    """
    Saves the rects_modest data to a .feather file.
    Args:
        rects_modest (list of dict): List of dictionaries containing bounding box data.
        savepath (str): Path to save the .feather file.
        frame_id (int): {Frame ID}.feather
        logger (logging.Logger): Logger object.
    """
    data = {
        "box_center_x": [rect['box_center'][0] for rect in rects_modest],
        "box_center_y": [rect['box_center'][1] for rect in rects_modest],
        "box_length": [rect['box_length'] for rect in rects_modest],
        "box_width": [rect['box_width'] for rect in rects_modest],
        "ry": [rect['ry'] for rect in rects_modest]
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Dataframe size before saving: {df.shape}")
    save_path = os.path.join(savepath, frame_id)
    df.to_feather(save_path)

def process_frame(frame: str, scene_path: str, output_dir: str, logger: logging.Logger):
    """
    Process a single frame to estimate bounding boxes.
    Args:
        frame (str): Frame name.
        scene_path (str): Path to the current scene.
        output_dir (str): Output directory.
        logger (logging.Logger): Logger object.
    """
    try:
        frame_path = os.path.join(scene_path, frame) # test_ge_script/scene_id/frame_id.feather
        
        save_dir = os.path.join(output_dir, os.path.basename(scene_path)) # test_bbox_script/scene_id
        os.makedirs(save_dir, exist_ok=True)
        
        # non_ground_file = os.path.join(frame_path, 'non_ground.feather')
        non_ground_points = pd.read_feather(frame_path)
        
        bbox_estimator = bboxer.Bboxer()
        bbox_estimator.cluster(non_ground_points.to_numpy()[:, :2])
        rects = bbox_estimator.estimate_bboxes_from_clusters_modest(
            bbox_estimator.clustered_points,
            bbox_estimator.clustered_labels, 
            'closeness_to_edge'
        )
        
        logger.info(f"Processing frame {frame}: Found {len(rects)} bounding boxes")
        
        if os.access(save_dir, os.W_OK):
            save_bboxes_to_feather(rects, save_dir, frame, logger)
        else:
            logger.error(f"Cannot write to directory {save_dir}")
        
        return frame
    except Exception as e:
        logger.error(f"Error processing frame {frame}: {str(e)}")
        return None

def process_scene(scene_tuple):
    """
    Process a single scene sequentially.
    Args:
        scene_tuple (tuple): Contains (scene_path, output_dir, logger)
    Returns:
        list: List of processed frames
    """
    scene_path, output_dir, logger = scene_tuple
    frames_in_scene = []
    
    try:
        logger.info(f"Processing scene: {os.path.basename(scene_path)}")
        
        # Process frames sequentially
        for frame in os.listdir(scene_path):
            result = process_frame(frame, scene_path, output_dir, logger)
            if result is not None:
                frames_in_scene.append(result)
            
        logger.info(f"Completed scene {os.path.basename(scene_path)}: {len(frames_in_scene)} frames processed")
        return frames_in_scene
    
    except Exception as e:
        logger.error(f"Error processing scene {scene_path}: {str(e)}")
        return frames_in_scene

def main():
    home = os.path.expanduser("~")
    ge_data_dir = os.path.join(home, 'buni', 'output-data', 'av2', 'test_ge_script')
    output_dir = os.path.join(home, 'buni', 'output-data', 'av2', 'test_bbox_script')
    logger = get_logger()
    
    start_time = time.time()
    
    # Create list of scene tuples for parallel processing
    scene_tuples = [
        (os.path.join(ge_data_dir, scene), output_dir, logger)
        for scene in os.listdir(ge_data_dir)
    ]
    
    scene_count = 10 # CONFIG PARAMETER: Number of scenes to process
    scene_tuples = scene_tuples[:scene_count]
    # Process scenes in parallel with non-daemon processes
    with Pool(processes=max(1, cpu_count() - 1), maxtasksperchild=1) as pool:
        scene_frame_list = pool.map(process_scene, scene_tuples)
    
    total_frames = sum(len(frames) for frames in scene_frame_list)
    logger.info(f"Processing complete:")
    logger.info(f"Number of scenes processed: {len(scene_frame_list)}")
    logger.info(f"Total frames processed: {total_frames}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()