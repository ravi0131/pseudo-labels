import numpy as np
import pandas as pd
from config import CONFIG
from stats_filter2 import apply_rect_filter
import os

def grid_search_rect_filter(input_path, ar_range, area_range,base_save_path ,config):
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
                    narrow_boxes_df, other_boxes_df = apply_rect_filter(df, 'aspect_ratio', 'area', aspect_ratio, area)
                    
                    # print(selected_boxes_df.head())
                    # print(discarded_boxes_df.head())
                    save_path = os.path.join(base_save_path, ar_dir_name, area_dir_name, scene, frame_id)
                    # print(save_path)
                    os.makedirs(save_path, exist_ok=True)
                    
                    narrow_boxes_df.to_feather(os.path.join(save_path, "narrow_boxes.feather"))
                    other_boxes_df.to_feather(os.path.join(save_path, "other_boxes.feather"))

def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG['HOME_PATH'][CONFIG['OS']])
    if CONFIG['ROI']:
        ps_base_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['ROI'])
        grid_rect_save_path = os.path.join(home, *CONFIG['GRID_SEARCH_RECT']['PATH']['ROI'])
        
    else:
        ps_base_path = os.path.join(home, *CONFIG['BBOX_FILE_PATHS']['FULL_RANGE'])
        grid_rect_save_path = os.path.join(home, *CONFIG['GRID_SEARCH_RECT']['PATH']['FULL_RANGE'])
    
    aspect_ratio_config = CONFIG['GRID_SEARCH_RECT']['ASPECT_RATIO']
    area_config = CONFIG['GRID_SEARCH_RECT']['AREA']
    rect_ar_range = np.arange(*aspect_ratio_config['RANGE'], aspect_ratio_config['STEP'])
    rect_area_range = np.arange(*area_config['RANGE'], area_config['STEP'])
    
    grid_search_rect_filter(ps_base_path, rect_ar_range, rect_area_range, grid_rect_save_path, CONFIG)




if __name__ == "__main__":
    main()
