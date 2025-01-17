import av2.datasets.sensor as sensor
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import av2.structures.sweep as sweep
import ge_av2.ground_estimator as ground_exorciser
import ge_av2.utilities as utils
import pandas as pd
import os
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time  # Import time module
from config import CONFIG
def process_frame(frame_path, output_dir, scene_id, dataset_path):
    # frame_dir = os.path.join(output_dir, scene_id, f"{frame_path.stem}.feather")
    # os.makedirs(frame_dir, exist_ok=True)

    lidar_frame = sweep.Sweep.from_feather(frame_path)
    if CONFIG['ROI']:
        points_roi = utils.filter_points_in_ROI(lidar_frame.xyz, **CONFIG['GE_RANGE'])
    # points_roi = utils.filter_points_in_ROI(lidar_frame.xyz, x_range=(0, 40), y_range=(-20, 20))
    points_roi = lidar_frame.xyz
    _, non_ground, _ = ground_exorciser.remove_ground(lidar_frame.xyz, points_roi)

    # Save non-ground points
    non_ground_path = os.path.join(output_dir, scene_id, f"{frame_path.stem}.feather")
    non_ground_df = pd.DataFrame(non_ground, columns=['x', 'y', 'z'])
    non_ground_df.to_feather(non_ground_path)

    return frame_path.stem

def process_scene(scene_id, dataset_path, output_dir):
    dataset = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)
    scene_dir = os.path.join(output_dir, scene_id)
    os.makedirs(scene_dir, exist_ok=True)

    lidar_frames = dataset.get_ordered_log_lidar_fpaths(scene_id)
    process_frame_partial = partial(process_frame, 
                                  output_dir=output_dir,
                                  scene_id=scene_id,
                                  dataset_path=dataset_path)

    with mp.Pool() as pool:
        results = pool.map(process_frame_partial, lidar_frames)

    return results

def main():
    # Record the start time
    start_time = time.time()

        # Setup paths
    #NOTE: Change the base path as needed
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    dataset_path = Path(os.path.join(home, *CONFIG['AV2_DATASET_PATH']))
    if CONFIG['ROI']:
        output_dir = os.path.join(home, *CONFIG['GROUND_ESTIMATION_FILE_PATHS']['ROI'])
    else:
        output_dir = os.path.join(home, *CONFIG['GROUND_ESTIMATION_FILE_PATHS']['FULL_RANGE'])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dataset and get scene IDs
    dataset = AV2SensorDataLoader(data_dir=dataset_path, labels_dir=dataset_path)
    scene_ids = dataset.get_log_ids()
    scene_count = 10  # CONFIG PARAMETER: Number of scenes to process
    scene_ids = scene_ids[:scene_count]
    print(f"Processing {len(scene_ids)} scenes")

    # Process each scene
    for scene_id in scene_ids:
        print(f"Processing scene: {scene_id}")
        processed_frames = process_scene(scene_id, dataset_path, output_dir)
        print(f"Completed {len(processed_frames)} frames for scene {scene_id}")

    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
