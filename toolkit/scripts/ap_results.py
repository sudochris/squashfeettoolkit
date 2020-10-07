import numpy as np

from os import path
from sklearn.metrics import average_precision_score
import pandas as pd

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.utils import file as file_utils
from toolkit.utils.timer import timed


@timed
def ap_results(app_settings:ApplicationSettings, datasets):
    output_file = file_utils.join_folders([app_settings.output_folder(), "Tab6_average_precisions.csv"])
    if file_utils.exists(output_file):
        logger.debug(f"Skipping {output_file}. File already exists")
        return

    column_names = ["dataset", "algorithm", "matching_type"]
    column_names.extend([f"{i}px" for i in np.arange(5, 51, 5)])
    ap_result = pd.DataFrame(columns=column_names)

    new_row = {}
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        logger.info(f"Processing {dataset_name}")
        result_folder = dataset.result_folder
        new_row.update({"dataset": dataset_name})
        for algorithm_type in ["A1F0_openpose_body25", "A1F1_openpose_coco", "A1F2_openpose_mpi", "A2_posenet"]:
            new_row.update({"algorithm": algorithm_type})
            for matching_type in ["MATCH_ALL", "MATCH_INDIVIDUAL"]:
                new_row.update({"matching_type": matching_type})
                ma_file = path.join(result_folder, f"{algorithm_type}.{matching_type}.markerresult.csv")
                data_table = pd.read_csv(ma_file)

                for thresh_num in np.arange(1, 11):
                    filtered = data_table[data_table["thresh_num"] == thresh_num]

                    y_true = filtered["tp"]
                    y_scores = filtered["confidence"]
                    ap = average_precision_score(y_true, y_scores)
                    new_row.update({f"{thresh_num*5}px": ap})

                ap_result = ap_result.append(new_row, ignore_index=True)
    ap_result.to_csv(output_file)