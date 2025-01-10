CONFIG = {
    "ROI": False,
    "GE_RANGE": {
        "x_range": (0, 40),
        "y_range": (-20, 20)
    },
    "SCENE_COUNT": 10,
    
    "GE_EXPORT_DIR": "ge_fullrange",
    "BBOX_EXPORT_DIR": "bbox_fullrange",
    
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