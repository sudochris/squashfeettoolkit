from scipy import stats
import pandas as pd
import numpy as np

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.datamodel.matching import FrameExperiments
from toolkit.utils import file as file_utils
from toolkit.utils.timer import timed


@timed
def frame_type_significance(app_settings: ApplicationSettings, datasets):
    p_thresh = 0.05
    test_output_file = file_utils.join_folders([app_settings.output_folder(), "t-test_results.csv"])
    if file_utils.exists(test_output_file):
        logger.debug(f"Skipping {test_output_file}. File already exists")
        return

    test_result = pd.DataFrame()
    test_result_row = {}

    logger.info("{:^20} | {:^8} | {:^12} | {:^12} | {:^8} | {:^8}".format(
        "dataset", "metric", "frame_type_a", "frame_type_b", "p", f"p < {p_thresh}"))

    logger.info("{:->{}}".format("", 28*3))
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        test_result_row.update({"dataset": dataset_name})
        for metric in ["PPV", "TPR", "TS", "F1"]:
            test_result_row.update({"metric": metric})

            data_matrix = np.zeros((11 * 5, 3))
            for frame_experiment in FrameExperiments:
                frame_experiment_name = frame_experiment.name
                input_filename = f"FrameMeasures.MATCH_ALL.{frame_experiment_name}.csv"
                input_filepath = file_utils.join_folders([dataset.result_folder, input_filename])

                input_table = pd.read_csv(input_filepath)

                a0 = input_table[f"A0_arttrack_{metric}"]
                a1f0 = input_table[f"A1F0_openpose_body25_{metric}"]
                a1f1 = input_table[f"A1F1_openpose_coco_{metric}"]
                a1f2 = input_table[f"A1F2_openpose_mpi_{metric}"]
                a2 = input_table[f"A2_posenet_{metric}"]

                fall_col = np.vstack((a0.values, a1f0.values))
                fall_col = np.vstack((fall_col, a1f1.values))
                fall_col = np.vstack((fall_col, a1f2.values))
                fall_col = np.vstack((fall_col, a2.values))
                fall_col = fall_col.ravel()

                data_matrix[:, frame_experiment] = fall_col

            # TEST
            fall_fral = stats.ttest_ind(data_matrix[:, FrameExperiments.ALL_FRAMES], data_matrix[:, FrameExperiments.RALLY_ONLY])
            fall_fnon = stats.ttest_ind(data_matrix[:, FrameExperiments.ALL_FRAMES], data_matrix[:, FrameExperiments.NONRALLY_ONLY])
            fral_fnon = stats.ttest_ind(data_matrix[:, FrameExperiments.RALLY_ONLY], data_matrix[:, FrameExperiments.NONRALLY_ONLY])

            logger.info("{:^20} | {:^8} | {:^12} | {:^12} | {:>8.2f} | {:^8}".format(
                dataset_name, metric, "FA", "FR", fall_fral.pvalue, "yes" if fall_fral.pvalue < p_thresh else "no"))
            logger.info("{:^20} | {:^8} | {:^12} | {:^12} | {:>8.2f} | {:^8}".format(
                dataset_name, metric, "FA", "FN", fall_fnon.pvalue, "yes" if fall_fnon.pvalue < p_thresh else "no"))
            logger.info("{:^20} | {:^8} | {:^12} | {:^12} | {:>8.2f} | {:^8}".format(
                dataset_name, metric, "FR", "FN", fral_fnon.pvalue, "yes" if fral_fnon.pvalue < p_thresh else "no"))

            test_result_row.update({"frame_type_a": "FA", "frame_type_b": "FR", "p": fall_fral.pvalue})
            test_result = test_result.append(test_result_row, ignore_index=True)

            test_result_row.update({"frame_type_a": "FA", "frame_type_b": "FN", "p": fall_fnon.pvalue})
            test_result = test_result.append(test_result_row, ignore_index=True)

            test_result_row.update({"frame_type_a": "FR", "frame_type_b": "FN", "p": fral_fnon.pvalue})
            test_result = test_result.append(test_result_row, ignore_index=True)
    test_result.to_csv(test_output_file)
