import cv2 as cv
import pandas as pd
from toolkit.datamodel.bodyparts import ArtTrack, OpenPoseBody25, OpenPoseCoco, OpenPoseMpi, PoseNet
from toolkit.datamodel.dataset import load_annotation_data, load_detection_data
from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.utils import frame as f_utils
from toolkit.utils import marker as m_utils
from toolkit.utils import file as file_utils
from toolkit.utils.configuration import AlgorithmType
from toolkit.utils.timer import timed


@timed
def overall_statistics(app_settings: ApplicationSettings, datasets):
    frame_results = pd.DataFrame()
    marker_results = pd.DataFrame()
    frame_results_row = {}
    marker_results_row = {}
    
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        frame_events = f_utils.FrameEvents(dataset.events_file)
        num_frames = dataset.num_frames
#        num_frames = int(cv.VideoCapture(dataset.video_file).get(cv.CAP_PROP_FRAME_COUNT))
        print(f"{dataset_name} = {num_frames}")
        frame_results_row.update({"video": dataset_name})
        marker_results_row.update({"video": dataset_name})

        rally_frames = 0
        nonrally_frames = 0
        for i in range(num_frames):
            is_rally = frame_events.last_group_event_change_for(i, "game_state") == "rally_start"
            if is_rally:
                rally_frames += 1
            else:
                nonrally_frames += 1

        annotation_data = load_annotation_data(dataset)
        valid_markers = m_utils.filter_markers_by_location(annotation_data.markers)

        total = rally_frames + nonrally_frames
        frame_results_row.update({"rally_frames":             rally_frames})
        frame_results_row.update({"rally_frames_perc":        rally_frames / total})
        frame_results_row.update({"non_rally_frames":         nonrally_frames})
        frame_results_row.update({"non_rally_frames_perc":    nonrally_frames / total})
        frame_results_row.update({"total_frames":             total})
        frame_results_row.update({"annotations":              valid_markers.shape[0]})
        frame_results_row.update({"annotations_per_frame":    valid_markers.shape[0] / num_frames})

        feet_map = {AlgorithmType.A0_ARTTRACK:          [ArtTrack.LAnkle, ArtTrack.RAnkle],
                    AlgorithmType.A1F0_OPENPOSE_BODY25: [OpenPoseBody25.LHeel, OpenPoseBody25.RHeel],
                    AlgorithmType.A1F1_OPENPOSE_COCO:   [OpenPoseCoco.LAnkle, OpenPoseCoco.RAnkle],
                    AlgorithmType.A1F2_OPENPOSE_MPI:    [OpenPoseMpi.LAnkle, OpenPoseMpi.RAnkle],
                    AlgorithmType.A2_POSENET:           [PoseNet.LAnkle, PoseNet.RAnkle]}

        marker_results_row.update({"annotations": valid_markers.shape[0]})
        for algorithm in dataset.algorithms:
            algorithm_name = algorithm.algorithm_name
            d_mids = feet_map[algorithm.algorithm_type]
            detection_data = load_detection_data(dataset, algorithm)  # Load detection data
            detection_markers = detection_data.markers
            detection_filtered = m_utils.filter_markers_by_location(detection_markers)
            detection_filtered = m_utils.filter_markers_by_mids(detection_filtered, d_mids)
            marker_results_row.update({algorithm_name: detection_filtered.shape[0]})

        logger.debug(marker_results_row)
        logger.debug(frame_results_row)

        marker_results = marker_results.append(marker_results_row, ignore_index=True)
        frame_results = frame_results.append(frame_results_row, ignore_index=True)

    frame_output_file = file_utils.join_folders([app_settings.output_folder(), "Tab3_dataset_overview.csv"])
    marker_output_file = file_utils.join_folders([app_settings.output_folder(), "Tab4_annotation_overview.csv"])

    if not file_utils.exists(frame_output_file):
        frame_results.to_csv(frame_output_file)
    else:
        logger.debug(f"Skipping {frame_output_file}. File already exists")

    if not file_utils.exists(marker_output_file):
        marker_results.to_csv(marker_output_file)
    else:
        logger.debug(f"Skipping {marker_output_file}. File already exists")

