import numpy as np

from toolkit.datamodel import classification
from toolkit.datamodel.dataset import load_annotation_data, load_detection_data
from toolkit.exporter.results import FrameResult, MarkerResult
from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.datamodel.matching import Experiments, get_matcher_for_experiment
from toolkit.datamodel.rendering import Rendering
from toolkit.utils.timer import timed
from toolkit.utils import frame as f_utils
from toolkit.utils import marker as m_utils
from toolkit.utils import color as c_utils
from toolkit.utils import file as file_utils


@timed
def match_experiments(app_settings: ApplicationSettings, datasets):

    logger.info("{:^32} | {:^32} | {:^32}".format("matching_type", "dataset", "algorithm"))
    logger.info("{:->{}}".format("", 34*3))
    for selected_experiment in Experiments:
        selected_experiment_name = selected_experiment.name

        experiment_matcher = get_matcher_for_experiment(selected_experiment)
        rendering = Rendering(app_settings.render(), 1)

        for dataset in datasets:
            dataset_name = dataset.dataset_name
            annotation_data = load_annotation_data(dataset)
            max_frame = max(annotation_data.markers[:, 0])

            frame_events = f_utils.FrameEvents(dataset.events_file)

            rendering.set_video_file(dataset.video_file)

            i_size = (annotation_data.framesize[0], annotation_data.framesize[1])

            for algorithm in dataset.algorithms:
                algorithm_name = algorithm.algorithm_name

                detection_data = load_detection_data(dataset, algorithm)
                experiment_instance = experiment_matcher[algorithm.algorithm_type]

                rendering.reset_video_file()
                algorithm_classification_result = {"TP": 0, "FP": 0, "FN": 0}

                thresholds = np.arange(0, 51, 5) / i_size[0]

                algorithm_frame_result = FrameResult(int(max_frame), len(thresholds))
                algorithm_marker_result = MarkerResult()

                frame_result_filename = \
                    file_utils.join_folders([dataset.result_folder,
                                             f"{algorithm_name}.{selected_experiment_name}.frameresult.csv"])

                marker_result_filename = \
                    file_utils.join_folders([dataset.result_folder,
                                             f"{algorithm_name}.{selected_experiment_name}.markerresult.csv"])

                if file_utils.exists(frame_result_filename) and file_utils.exists(marker_result_filename):
                    logger.debug(f"Skipping {frame_result_filename} and {marker_result_filename}. Files already exist")
                    continue

                logger.info("{:^32} | {:^32} | {:^32}".format(selected_experiment_name, dataset_name, algorithm_name))

                for frame_pos in range(int(max_frame)):

                    rendering.read_next_frame()
                    rendering.add_overlay_text(f"{dataset_name=}")
                    rendering.add_overlay_text(f"{algorithm_name=}")
                    rendering.add_overlay_text(f"{frame_pos=}")
                    rendering.add_overlay_text()

                    # 0. Schritt: Erfahre ob es sich um einen frame handelt der einen ballwechsel zeigt
                    last_game_state_event = frame_events.last_group_event_change_for(frame_pos, "game_state")
                    frame_is_rally = last_game_state_event == "rally_start"

                    # 1. Filter: Marker für aktuelle frame position
                    a_markers_in_frame = m_utils.filter_markers_by_frame(annotation_data.markers, frame_pos)
                    d_markers_in_frame = m_utils.filter_markers_by_frame(detection_data.markers, frame_pos)

                    # 2. Filter: Marker außerhalb des bildes
                    a_markers_by_location = m_utils.filter_markers_by_location(a_markers_in_frame)
                    d_markers_by_location = m_utils.filter_markers_by_location(d_markers_in_frame)

                    for thresh_num, thresh in enumerate(thresholds):
                        frame_classification_result = {"TP": 0, "FP": 0, "FN": 0}

                        for wanted_d_mid, wanted_a_mids in experiment_instance.items():
                            # 2. Filter: Marker des entsprechenden Typs
                            a_markers_by_mid = m_utils.filter_markers_by_mids(a_markers_by_location, wanted_a_mids)
                            d_markers_by_mid = m_utils.filter_markers_by_mids(d_markers_by_location, [wanted_d_mid])

                            a_markers_filtered = a_markers_by_mid
                            d_markers_filtered = d_markers_by_mid

                            real_result = classification.classify_norm(d_markers_filtered, a_markers_filtered, thresh)

                            for is_tp, label in zip([True, False], ["TP", "FP"]):
                                for tps in real_result[label]:
                                    res_uid, res_mid, res_uv, res_distance, _, res_c = tps
                                    res_norm_err = res_distance
                                    res_px_err = res_norm_err * i_size[0]
                                    algorithm_marker_result.add_data_for(frame_pos, thresh_num, res_mid, is_tp,
                                                                         res_px_err, res_norm_err, res_c,
                                                                         frame_is_rally)

                            partial_classification = {
                                "TP": len(real_result["TP"]),
                                "FP": len(real_result["FP"])
                            }

                            for k in ["TP", "FP"]:
                                frame_classification_result[k] += partial_classification[k]

                            for a_marker in a_markers_filtered:
                                _, _, a_mid, a_uv, a_c, a_uid = m_utils.unpack_marker(a_marker)
                                a_scaled_marker = m_utils.scale_and_round(a_uv, i_size)
                                rendering.draw_annotation_marker(a_scaled_marker,
                                                                 c_utils.annotation_color_by_mid(a_mid),
                                                                 c_utils.annotation_marker_type_by_mid(a_mid))

                            for d_marker in d_markers_filtered:
                                _, _, d_mid, d_uv, d_c, d_uid = m_utils.unpack_marker(d_marker)
                                d_scaled_marker = m_utils.scale_and_round(d_uv, i_size)
                                rendering.draw_detection_marker(d_scaled_marker,
                                                                c_utils.algorithm_color_by_mid(d_mid, algorithm.algorithm_type),
                                                                c_utils.algorithm_marker_type_by_mid(d_mid, algorithm.algorithm_type))

                        frame_fn = len(a_markers_by_location) - frame_classification_result["TP"]
                        if frame_fn > 0:
                            frame_classification_result["FN"] = frame_fn

                        algorithm_frame_result.set_data_for(frame_pos, thresh_num, frame_classification_result["TP"],
                                                            frame_classification_result["FP"],
                                                            frame_classification_result["FN"])

                        rendering.add_overlay_text(f"@{thresh_num=}*{frame_classification_result=}")
                        rendering.add_overlay_text(f"@{thresh_num=}*{algorithm_classification_result=}")

                    action = rendering.render(True)
                    if action == Rendering.Action.EXIT:
                        break

                    algorithm_frame_result.set_is_rally_for_frame(frame_pos, frame_is_rally)

                algorithm_frame_result.save(frame_result_filename)
                algorithm_marker_result.save(marker_result_filename)
