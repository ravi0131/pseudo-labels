import os
import pandas as pd
from config import CONFIG
from stats_filter2 import apply_large_sq_filter
import numpy as np


def grid_search_square_filter(input_path, ar_range, area_range,base_save_path ,config):
    # print(f"Processing {input_frame_path}")
    aspect_ratio_dir = config['GRID_SEARCH_RECT']['AR_THRESHOLD_BASE_DIR_NAME']
    area_dir = config['GRID_SEARCH_RECT']['AREA_THRESHOLD_BASE_DIR_NAME']
    for _, aspect_ratio in enumerate(ar_range):
        ar_dir_name = f"{aspect_ratio_dir}_{aspect_ratio:.1f}"
        
        for _, area in enumerate(area_range):
            area_dir_name = f"{area_dir}_{area}"
            
            for scene in os.listdir(input_path):
                scene_path = os.path.join(input_path, scene)    
                
                for frame in os.listdir(scene_path):
                    frame_id = frame.split(".")[0]
                    frame_path = os.path.join(scene_path, frame)
                    
                    # print(f"Processing {frame_path}")
                    # print(f"Aspect Ratio: {aspect_ratio}, Area: {area}")
                    df = pd.read_feather(frame_path)
                    
                    df['aspect_ratio'] = df['box_width'] /df['box_length']
                    df['area'] = df['box_width'] * df['box_length']
                    large_squares_df, rest_boxes_df = apply_large_sq_filter(df, 'aspect_ratio', 'area', aspect_ratio, area)
                    
                    # print(large_squares_df.head())
                    # print(rest_boxes_df.head())
                    save_path = os.path.join(base_save_path, ar_dir_name, area_dir_name, scene, frame_id)
                    # print(save_path)
                    os.makedirs(save_path, exist_ok=True)
                    
                    large_squares_df.to_feather(os.path.join(save_path, "large_squares.feather"))
                    rest_boxes_df.to_feather(os.path.join(save_path, "rest_boxes.feather"))


def main():
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
    
    grid_search_square_filter(ps_base_path, sq_ar_range, sq_area_range, grid_sq_save_path, CONFIG)

if __name__ == '__main__':
    main()