from enum import IntEnum
from os import path

import cv2 as cv
import numpy as np

from toolkit.datamodel.bodyparts import Annotation, ArtTrack, OpenPoseBody25, OpenPoseCoco, OpenPoseMpi, PoseNet
from toolkit.datamodel.dataset import load_annotation_data, load_detection_data
from toolkit.heatmap import transformations
from toolkit.heatmap.accumulator import Accumulator
from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger

from toolkit.utils.configuration import AlgorithmType
from toolkit.utils import marker as m_utils
from toolkit.utils import file as file_utils
import toolkit.heatmap.calibration as calibration
from toolkit.utils.timer import timed


class AccumulatorType(IntEnum):
    ANNOTATION = 0,
    DETECTIONS = 1


def generate_heatmaps_for_dataset(dataset):

    dataset_name = dataset.dataset_name
    annotation_data = load_annotation_data(dataset)
    num_frames = max(annotation_data.markers[:, 0])

    (I_COLS, I_ROWS) = annotation_data.framesize

    camera_accumulators = {
        AccumulatorType.ANNOTATION: Accumulator(I_ROWS, I_COLS),
        AccumulatorType.DETECTIONS: {}
    }

    top_down_accumulators = {
        AccumulatorType.ANNOTATION: Accumulator(975, 640),
        AccumulatorType.DETECTIONS: {}
    }

    annotation_mids = [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]

    mids = {
        AlgorithmType.A0_ARTTRACK:          [ArtTrack.RAnkle, ArtTrack.LAnkle],
        AlgorithmType.A1F0_OPENPOSE_BODY25: [OpenPoseBody25.RHeel, OpenPoseBody25.LHeel],
        AlgorithmType.A1F1_OPENPOSE_COCO:   [OpenPoseCoco.RAnkle, OpenPoseCoco.LAnkle],
        AlgorithmType.A1F2_OPENPOSE_MPI:    [OpenPoseMpi.RAnkle, OpenPoseMpi.LAnkle],
        AlgorithmType.A2_POSENET:           [PoseNet.RAnkle, PoseNet.LAnkle],
    }

    calibration_data = calibration.load_from_file(dataset.calibration_file)

    logger.info(f"[{dataset_name}] Processing annotation data")
    for frame_pos in range(int(num_frames)):
        a_markers_in_frame = m_utils.filter_markers_by_frame(annotation_data.markers, frame_pos)
        a_markers_by_location = m_utils.filter_markers_by_location(a_markers_in_frame)
        a_markers_filtered = a_markers_by_location

        for a_marker in a_markers_filtered:
            f_pos, a_pid, a_mid, a_uv, a_c, a_uid = m_utils.unpack_marker(a_marker)
            if a_mid in annotation_mids:
                a_px = a_uv * (I_COLS, I_ROWS)
                camera_accumulators[AccumulatorType.ANNOTATION].add_point(int(a_px[1]), int(a_px[0]))

                a_wp = calibration.estimate_from_pixel(calibration_data, [a_px[0] + 200, a_px[1]])
                x_hm = int(np.interp(a_wp[0], [0, 6.4], [0, 640], 0, 640))
                z_hm = int(np.interp(a_wp[2], [0, 9.75], [0, 975], 0, 975))
                top_down_accumulators[AccumulatorType.ANNOTATION].add_point(z_hm, x_hm)


    for algorithm in dataset.algorithms:
        algorithm_type = algorithm.algorithm_type
        algorithm_name = algorithm.algorithm_name
        logger.info(f"[{dataset_name}] Processing {algorithm_name}")

        camera_accumulators[AccumulatorType.DETECTIONS].update({algorithm_type: Accumulator(I_ROWS, I_COLS)})
        top_down_accumulators[AccumulatorType.DETECTIONS].update({algorithm_type: Accumulator(975, 640)})

        detection_data = load_detection_data(dataset, algorithm)    # Load detection data

        for frame_pos in range(int(num_frames)):
            d_markers_in_frame = m_utils.filter_markers_by_frame(detection_data.markers, frame_pos)
            d_markers_by_location = m_utils.filter_markers_by_location(d_markers_in_frame)
            d_markers_filtered = d_markers_by_location
            for d_marker in d_markers_filtered:
                f_pos, d_pid, d_mid, d_uv, d_c, d_uid = m_utils.unpack_marker(d_marker)
                if d_mid in mids[algorithm_type]:
                    d_px = d_uv * (I_COLS, I_ROWS)
                    camera_accumulators[AccumulatorType.DETECTIONS][algorithm_type].add_point(int(d_px[1]), int(d_px[0]))

                    d_wp = calibration.estimate_from_pixel(calibration_data, [d_px[0] + 200, d_px[1]])
                    x_hm = int(np.interp(d_wp[0], [0, 6.4], [0, 640], 0, 640))
                    z_hm = int(np.interp(d_wp[2], [0, 9.75], [0, 975], 0, 975))
                    top_down_accumulators[AccumulatorType.DETECTIONS][algorithm_type].add_point(z_hm, x_hm)

    return camera_accumulators, top_down_accumulators

@timed
def generate_heatmaps(app_settings:ApplicationSettings, datasets):

    def transform_and_save(dataset_name, view_type, algorithm_name, accumulator):
        def transform_accumulator_to_heatmap(accumulator):
            heatmap = transformations.logarithmic(accumulator)
            return heatmap

        def save_heatmap(dataset_name, view_type, algorithm_name, image):
            file_name = f"{dataset_name}_{view_type}_{algorithm_name}.png"
            heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "gray_heatmaps"])
            file_utils.make_dirs(heatmaps_folder)

            file_path = file_utils.join_folders([heatmaps_folder, file_name])
            if not file_utils.exists(file_path):
                cv.imwrite(file_path, image)
            else:
                logger.debug(f"Skipping {file_path}. File already exists")

        heatmap = transform_accumulator_to_heatmap(accumulator)
        save_heatmap(dataset_name, view_type, algorithm_name, heatmap)

    for dataset in datasets:
        dataset_name = dataset.dataset_name

        camera_accumulators, top_down_accumulators = generate_heatmaps_for_dataset(dataset)

        camera_accumulator_annotation = camera_accumulators[AccumulatorType.ANNOTATION]
        camera_accumulator_detections = camera_accumulators[AccumulatorType.DETECTIONS]


        top_down_accumulator_annotation = top_down_accumulators[AccumulatorType.ANNOTATION]
        top_down_accumulator_detections = top_down_accumulators[AccumulatorType.DETECTIONS]

        transform_and_save(dataset_name, "camera", "annotation", camera_accumulator_annotation)
        transform_and_save(dataset_name, "topdown", "annotation", top_down_accumulator_annotation)

        for algorithm_type in AlgorithmType:
            if algorithm_type == AlgorithmType.UNKNOWN:
                continue

            camera_accumulator_detection = camera_accumulator_detections[algorithm_type]
            top_down_accumulator_detection = top_down_accumulator_detections[algorithm_type]

            transform_and_save(dataset_name, "camera", algorithm_type, camera_accumulator_detection)
            transform_and_save(dataset_name, "topdown", algorithm_type, top_down_accumulator_detection)
