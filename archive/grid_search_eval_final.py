import os
import pandas as pd
from config import CONFIG
def calculate_total_metrics(base_save_path: str):
    """
    Calculate the total metrics (mAP, precision, recall) for all scenes
    under base_save_path, for both IoU=0.3 and IoU=0.5.
    
    Returns:
        (
          mAP_03, precision_03, recall_03,
          mAP_05, precision_05, recall_05
        )
    """
    # For IoU=0.3
    mAPs_03 = []
    precisions_03 = []
    recalls_03 = []
    
    # For IoU=0.5
    mAPs_05 = []
    precisions_05 = []
    recalls_05 = []
    
    for scene_id in os.listdir(base_save_path):
        scene_path = os.path.join(base_save_path, scene_id)
        if not os.path.isdir(scene_path):
            continue
        
        for frame_id in os.listdir(scene_path):
            frame_path = os.path.join(scene_path, frame_id)
            if not os.path.isdir(frame_path):
                continue
            
            # ---------- IoU=0.3 ----------
            results_file_03 = os.path.join(frame_path, "iou_0.3_.feather")
            if os.path.isfile(results_file_03):
                df_03 = pd.read_feather(results_file_03)
                mAPs_03.append(df_03["mAP"][0])
                precisions_03.append(df_03["precision"][0])
                recalls_03.append(df_03["recall"][0])
            
            # ---------- IoU=0.5 ----------
            results_file_05 = os.path.join(frame_path, "iou_0.5_.feather")
            if os.path.isfile(results_file_05):
                df_05 = pd.read_feather(results_file_05)
                mAPs_05.append(df_05["mAP"][0])
                precisions_05.append(df_05["precision"][0])
                recalls_05.append(df_05["recall"][0])
    
    # Safeguard: if lists are empty, return zeros
    if len(mAPs_03) == 0:
        mAP_03, precision_03, recall_03 = 0.0, 0.0, 0.0
    else:
        mAP_03 = sum(mAPs_03) / len(mAPs_03)
        precision_03 = sum(precisions_03) / len(precisions_03)
        recall_03 = sum(recalls_03) / len(recalls_03)
    
    if len(mAPs_05) == 0:
        mAP_05, precision_05, recall_05 = 0.0, 0.0, 0.0
    else:
        mAP_05 = sum(mAPs_05) / len(mAPs_05)
        precision_05 = sum(precisions_05) / len(precisions_05)
        recall_05 = sum(recalls_05) / len(recalls_05)
    
    return (
        mAP_03, precision_03, recall_03,
        mAP_05, precision_05, recall_05
    )

def collect_metrics_for_all_combinations(base_output_path: str):
    """
    Iterates over AR_x/AREA_y directories under `base_output_path`,
    computes total metrics for IoU=0.3 and IoU=0.5,
    and collects them into a list of dicts.

    Returns:
        A list of dicts. Each dict has:
         {
           'ar_dir': str,
           'area_dir': str,
           'mAP_03': float,
           'precision_03': float,
           'recall_03': float,
           'mAP_05': float,
           'precision_05': float,
           'recall_05': float
         }
    """
    combination_results = []

    for ar_dir in os.listdir(base_output_path):
        ar_path = os.path.join(base_output_path, ar_dir)
        if not os.path.isdir(ar_path):
            continue

        for area_dir in os.listdir(ar_path):
            area_path = os.path.join(ar_path, area_dir)
            if not os.path.isdir(area_path):
                continue

            (
                mAP_03, precision_03, recall_03,
                mAP_05, precision_05, recall_05
            ) = calculate_total_metrics(area_path)

            combination_results.append({
                'ar_dir': ar_dir,
                'area_dir': area_dir,
                'mAP_03': mAP_03,
                'precision_03': precision_03,
                'recall_03': recall_03,
                'mAP_05': mAP_05,
                'precision_05': precision_05,
                'recall_05': recall_05
            })

    return combination_results

def find_best_combinations(combination_results):
    """
    Finds combinations with:
      - highest mAP_03
      - highest precision_03
      - highest recall_03
      - highest mAP_05
      - highest precision_05
      - highest recall_05

    Returns a dict, e.g.:
    {
      'best_map_03': {...},
      'best_precision_03': {...},
      'best_recall_03': {...},
      'best_map_05': {...},
      'best_precision_05': {...},
      'best_recall_05': {...},
    }
    where each value is a single entry from combination_results
    that includes its 'ar_dir' and 'area_dir'.
    """
    if not combination_results:
        return {}

    best_map_03       = max(combination_results, key=lambda x: x['mAP_03'])
    best_precision_03 = max(combination_results, key=lambda x: x['precision_03'])
    best_recall_03    = max(combination_results, key=lambda x: x['recall_03'])

    best_map_05       = max(combination_results, key=lambda x: x['mAP_05'])
    best_precision_05 = max(combination_results, key=lambda x: x['precision_05'])
    best_recall_05    = max(combination_results, key=lambda x: x['recall_05'])

    return {
        'best_map_03': best_map_03,
        'best_precision_03': best_precision_03,
        'best_recall_03': best_recall_03,
        'best_map_05': best_map_05,
        'best_precision_05': best_precision_05,
        'best_recall_05': best_recall_05
    }

def main():
    home = os.path.join(os.path.expanduser("~"), CONFIG["HOME_PATH"][CONFIG["OS"]])
    filter_type = CONFIG['GRID_SEARCH_METRICS_MODE']['FILTER']
    print(f"Filter type: {filter_type} and roi: {CONFIG['ROI']}")
    if CONFIG["ROI"]:
        if filter_type == "rect_filter":
            base_output_path = os.path.join(home, *CONFIG['GRID_SEARCH_METRICS_PATH']['ROI']['RECT_FILTER_PATH'])
        else:  # square_filter
            base_output_path = os.path.join(home, *CONFIG['GRID_SEARCH_METRICS_PATH']['ROI']['SQUARE_FILTER_PATH'])
    else:
        if filter_type == "rect_filter":
            base_output_path = os.path.join(home, *CONFIG['GRID_SEARCH_METRICS_PATH']['FULL_RANGE']['RECT_FILTER_PATH'])
        else:  # square_filter
            base_output_path = os.path.join(home, *CONFIG['GRID_SEARCH_METRICS_PATH']['FULL_RANGE']['SQUARE_FILTER_PATH'])
    
    # 1) Collect the metrics for all (AR, AREA) combos, for both IoU=0.3 & IoU=0.5
    combination_results = collect_metrics_for_all_combinations(base_output_path)

    # 2) Identify which combos are best in each metric (six total)
    bests = find_best_combinations(combination_results)

    # 3) Print them out in a clear format
    if not bests:
        print("No combination results found.")
        return

    print("\n--- Results for IoU = 0.3 ---\n")
    print("Best mAP:")
    print(f"  AR dir:    {bests['best_map_03']['ar_dir']}")
    print(f"  AREA dir:  {bests['best_map_03']['area_dir']}")
    print(f"  mAP:       {bests['best_map_03']['mAP_03']:.4f}")
    print(f"  Precision: {bests['best_map_03']['precision_03']:.4f}")
    print(f"  Recall:    {bests['best_map_03']['recall_03']:.4f}\n")

    print("Best Precision:")
    print(f"  AR dir:    {bests['best_precision_03']['ar_dir']}")
    print(f"  AREA dir:  {bests['best_precision_03']['area_dir']}")
    print(f"  mAP:       {bests['best_precision_03']['mAP_03']:.4f}")
    print(f"  Precision: {bests['best_precision_03']['precision_03']:.4f}")
    print(f"  Recall:    {bests['best_precision_03']['recall_03']:.4f}\n")

    print("Best Recall:")
    print(f"  AR dir:    {bests['best_recall_03']['ar_dir']}")
    print(f"  AREA dir:  {bests['best_recall_03']['area_dir']}")
    print(f"  mAP:       {bests['best_recall_03']['mAP_03']:.4f}")
    print(f"  Precision: {bests['best_recall_03']['precision_03']:.4f}")
    print(f"  Recall:    {bests['best_recall_03']['recall_03']:.4f}\n")

    print("\n--- Results for IoU = 0.5 ---\n")
    print("Best mAP:")
    print(f"  AR dir:    {bests['best_map_05']['ar_dir']}")
    print(f"  AREA dir:  {bests['best_map_05']['area_dir']}")
    print(f"  mAP:       {bests['best_map_05']['mAP_05']:.4f}")
    print(f"  Precision: {bests['best_map_05']['precision_05']:.4f}")
    print(f"  Recall:    {bests['best_map_05']['recall_05']:.4f}\n")

    print("Best Precision:")
    print(f"  AR dir:    {bests['best_precision_05']['ar_dir']}")
    print(f"  AREA dir:  {bests['best_precision_05']['area_dir']}")
    print(f"  mAP:       {bests['best_precision_05']['mAP_05']:.4f}")
    print(f"  Precision: {bests['best_precision_05']['precision_05']:.4f}")
    print(f"  Recall:    {bests['best_precision_05']['recall_05']:.4f}\n")

    print("Best Recall:")
    print(f"  AR dir:    {bests['best_recall_05']['ar_dir']}")
    print(f"  AREA dir:  {bests['best_recall_05']['area_dir']}")
    print(f"  mAP:       {bests['best_recall_05']['mAP_05']:.4f}")
    print(f"  Precision: {bests['best_recall_05']['precision_05']:.4f}")
    print(f"  Recall:    {bests['best_recall_05']['recall_05']:.4f}")


if __name__ == "__main__":
    main()
