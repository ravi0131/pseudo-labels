CONFIG = {
    "ROI": True,
    "SCENE_COUNT": 10,
    "OS": "WINDOWS", # "LINUX" or "WINDOWS"
    
    "GE_RANGE": {
        "x_range": (0, 40),
        "y_range": (-20, 20)
    },
    "GT_LABELS_ROI": {
        "x_range": (0, 40),
        "y_range": (-20, 20)
    },
    
    "AV2_DATASET_PATH": ["dataset", "av2", "train"],
    
    "HOME_PATH":{
        "WINDOWS": "buni", 
        "LINUX": ""
    },
    
    "GROUND_ESTIMATION_FILE_PATHS":{
        "ROI": ["output-data", "av2", "ground_estimation", "roi"],
        "FULL_RANGE": ["output-data", "av2", "ground_estimation", "full_range"]
    },
    
    "BBOX_FILE_PATHS":{
        "ROI": ["output-data", "av2", "bboxes", "roi"],
        "FULL_RANGE": ["output-data", "av2", "bboxes", "full_range"]
    },
    
    "FILTERED_BBOX_FILE_PATHS":{
        "ROI": ["output-data", "av2", "filtered_bboxes", "roi"],
        "FULL_RANGE": ["output-data", "av2", "filtered_bboxes", "full_range"]
    },
    
    "METRICS_FILE_PATHS":{
        "ROI": {
            "NORMAL": ["output-data", "av2", "metrics", "roi", "normal"],
            "FILTERED": ["output-data", "av2", "metrics", "roi", "filtered"]
            },
        "FULL_RANGE": {
            "NORMAL": ["output-data", "av2", "metrics", "full_range", "normal"],
            "FILTERED": ["output-data", "av2", "metrics", "full_range", "filtered"]
            }
    },
    
    "ASPECT_RATIO_FILTER": {
        "min_ratio": 0.3,
        "max_ratio": 1.0
    },
    "AREA_FILTER_SQUARE": {
        "min_aspect_ratio": 0.6,
        "square_max_area": 1.0
    },
    "AREA_FILTER_RECT": {
        "max_aspect_ratio": 0.6,
        "max_area": 50
    },
     "NMS_IOU_THRESHOLD": 0.1,
}