CONFIG = {
    "ROI": True,
    "SCENE_COUNT": 10,
    "OS": "WINDOWS", # "LINUX" or "WINDOWS"
    
    "HOME_PATH":{
        "WINDOWS": "buni", 
        "LINUX": ""
    },
    
    "GE_RANGE": {
        "x_range": (0, 40),
        "y_range": (-20, 20)
    },
    "GT_LABELS_ROI": {
        "x_range": (0, 40),
        "y_range": (-20, 20)
    },
    
    "AV2_DATASET_PATH": ["dataset", "av2", "train"],
    
    
    
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
    
    "GRID_SEARCH_RECT":{
        "PATH":{
            "ROI":["output-data", "av2", "grid_search", "roi", "rect_filter"],
            "FULL_RANGE": ["output-data", "av2", "grid_search", "full_range", "rect_filter"],
        },
        "AR_THRESHOLD_BASE_DIR_NAME": "ar_threshold",
        "AREA_THRESHOLD_BASE_DIR_NAME": "area_threshold",
        
        "ASPECT_RATIO": {
            "RANGE":[0.1, 1],
            "STEP": 0.1,    
        },
        "AREA": {
            "RANGE":[10, 200],
            "STEP": 10,
        },
    
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

    "GT_CATEGORIES":[
        "REGULAR_VEHICLE",
        "LARGE_VEHICLE",
        "BUS",
        "TRUCK",
        "VEHICULAR_TRAILER",
        "TRUCK_CAB",
        "SCHOOL_BUS",
        "ARTICULATED_BUS",
        "RAILED_VEHICLE",   
    ]
}