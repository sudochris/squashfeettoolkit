from os import path
import pandas as pd
import numpy as np

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.datamodel.matching import Experiments, FrameExperiments
from toolkit.utils.timer import timed
from toolkit.utils import file as file_utils


@timed
def frameresult_metrics(app_settings:ApplicationSettings, datasets):
    """This script takes various *.frameresults.csv files and creates the corresponding FrameMeasures.* files"""
    for selected_frame_experiment in FrameExperiments:
        selected_frame_experiment_name = selected_frame_experiment.name

        for selected_experiment in Experiments:
            selected_experiment_name = selected_experiment.name

            for dataset in datasets:
                filename = f"FrameMeasures.{selected_experiment_name}.{selected_frame_experiment_name}.csv"
                output_file = file_utils.join_folders([dataset.result_folder, filename])
                if file_utils.exists(output_file):
                    logger.debug(f"Skipping {output_file}. File already exists")
                    continue

                num_thresh = len(np.arange(0, 51, 5))
                result_dataset = []
                result_header = ["px_rad"]

                for algorithm in dataset.algorithms:
                    algorithm_name = algorithm.algorithm_name
                    result_header.append(f"{algorithm_name}_PPV")
                    result_header.append(f"{algorithm_name}_TPR")
                    result_header.append(f"{algorithm_name}_TS")
                    result_header.append(f"{algorithm_name}_F1")

                for n_thresh in range(num_thresh):
                    result_row = [n_thresh * 5]
                    for algorithm in dataset.algorithms:
                        algorithm_name = algorithm.algorithm_name
                        filename = f"{algorithm_name}.{selected_experiment_name}.frameresult.csv"
                        algorithm_frame_result_file = path.join(dataset.result_folder, filename)
                        data = pd.read_csv(algorithm_frame_result_file)

                        # filter for isrally column
                        if selected_frame_experiment == FrameExperiments.RALLY_ONLY:
                            data = data[data["frame_is_rally"] == 1]
                        elif selected_frame_experiment == FrameExperiments.NONRALLY_ONLY:
                            data = data[data["frame_is_rally"] == 0]

                        data_sum = data.sum()
                        tp = data_sum[f"TP{n_thresh}"]
                        fp = data_sum[f"FP{n_thresh}"]
                        fn = data_sum[f"FN{n_thresh}"]

                        tpr = tp / (tp + fn)
                        ppv = tp / (tp + fp)
                        ts = tp / (tp + fn + fp)
                        f1 = (2 * (ppv * tpr)) / (ppv + tpr) if (ppv + tpr) > 0 else 0.0

                        result_row.append(ppv)
                        result_row.append(tpr)
                        result_row.append(ts)
                        result_row.append(f1)

                    result_dataset.append(result_row)

                pandas_result = pd.DataFrame(result_dataset, columns=result_header)

                pandas_result.to_csv(output_file)
