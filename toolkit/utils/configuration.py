import json as json
from os import path as path
from os import listdir as listdir
import os
from enum import Enum
from collections import namedtuple

from toolkit.utils.timer import timed

Algorithms = namedtuple("Algorithms", ["algorithm_name", "algorithm_type", "folder", "keypoint_file_pattern", "numpy_file", "full_keypoint_file_pattern"])
Dataset = namedtuple("Dataset", ["dataset_name", "court_image_file", "annotation_file", "annotation_numpy", "events_file", "calibration_file", "court_file", "num_frames", "video_file", "result_folder", "npz_folder", "algorithms"])

class AlgorithmType(Enum):
    UNKNOWN = "Unknown",
    A0_ARTTRACK = "A0",
    A1F0_OPENPOSE_BODY25 = "A1F0",
    A1F1_OPENPOSE_COCO = "A1F1",
    A1F2_OPENPOSE_MPI = "A1F2",
    A2_POSENET = "A2"

def infer_algorithm_type_from_name(algorithm_name):
    if algorithm_name == "A0_arttrack":
        return AlgorithmType.A0_ARTTRACK
    if algorithm_name == "A1F0_openpose_body25":
        return AlgorithmType.A1F0_OPENPOSE_BODY25
    if algorithm_name == "A1F1_openpose_coco":
        return AlgorithmType.A1F1_OPENPOSE_COCO
    if algorithm_name == "A1F2_openpose_mpi":
        return AlgorithmType.A1F2_OPENPOSE_MPI
    if algorithm_name == "A2_posenet":
        return AlgorithmType.A2_POSENET
    return AlgorithmType.UNKNOWN

@timed
def load_datasets(configuration_file):
    if not path.exists(configuration_file):
        print(f"Configuration file '{configuration_file}' does not exist!")
        exit(1)

    folder_prefix = path.dirname(path.abspath(configuration_file))
    datasets = []

    with open(configuration_file) as json_data_file:
        data = json.load(json_data_file)

        for dj in data["datasets"]:
            algorithms = []
            for aj in dj["algorithms"]:
                algorithm = Algorithms(
                    algorithm_name=aj["name"],
                    algorithm_type=infer_algorithm_type_from_name(aj["name"]),
                    folder=path.join(folder_prefix, aj["folder"]),
                    keypoint_file_pattern=path.join(folder_prefix, aj["keypoint_file_pattern"]),
                    numpy_file=path.join(folder_prefix, aj["numpy_file"]),
                    full_keypoint_file_pattern=path.join(folder_prefix, aj["folder"], aj["keypoint_file_pattern"])
                )
                algorithms.append(algorithm)

            dataset = Dataset(
                dataset_name=dj["name"],
                court_image_file=path.join(folder_prefix, dj["files"]["court_image_file"]),
                annotation_file=path.join(folder_prefix, dj["files"]["annotation_file"]),
                annotation_numpy=path.join(folder_prefix, dj["files"]["annotation_numpy"]),
                events_file=path.join(folder_prefix, dj["files"]["events_file"]),
                calibration_file=path.join(folder_prefix, dj["files"]["calibration_file"]),
                court_file=path.join(folder_prefix, dj["files"]["court_file"]),
                num_frames=dj["num_frames"],
                video_file=path.join(folder_prefix, dj["files"]["video_file"]),
                result_folder=path.join(folder_prefix, dj["files"]["result_folder"]),
                npz_folder=path.join(folder_prefix, dj["files"]["npz_folder"]),
                algorithms=algorithms
            )
            datasets.append(dataset)

    return datasets

def infer_filenames_for_algorithm(algorithm):
    num_files = len(listdir(algorithm.folder))

    for i in range(num_files):
        filename = algorithm.full_keypoint_file_pattern.format(i)
        if os.path.exists(filename):
            yield i, filename