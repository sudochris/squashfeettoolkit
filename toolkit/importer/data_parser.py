from toolkit.datamodel.bodyparts import Annotation, OpenPoseBody25, OpenPoseCoco, OpenPoseMpi, ArtTrack, PoseNet
from toolkit.logger import logger
import numpy as np
import json as json
from toolkit.datamodel.dataset import DetectionData as DetectionData

"""THIS FILE CONTAINS THE CONVERTER FUNCTIONS FOR JSON -> npy"""

def parse_annotation(file_name):
    result_data = DetectionData

    with open(file_name, 'r') as file:
        json_data = json.load(file)
        annotation_info = json_data["annotationInfo"]
        annotation_data = json_data["annotationData"]

        result_data.author = annotation_info["author"]
        frame_width = annotation_info["framewidth"]
        frame_height = annotation_info["frameheight"]

        result_data.markers = np.zeros((0, 6))

        for frame_info in annotation_data:
            frame_position = frame_info["framePosition"]
            marker_list = frame_info["pointsList"]

            for json_marker in marker_list:
                i = json_marker["pointid"]
                u = json_marker["u"]
                v = json_marker["v"]
                pid = 1 if i == Annotation.LFootP1 or i == Annotation.RFootP1 else 2
                result_data.markers = np.vstack([result_data.markers, [frame_position, pid, i, u, v, 1.0]])

        result_data.framesize = (frame_width, frame_height)

    return result_data


def parse_a0(file_generator, bodyparts):
    """ Parser for the arttrack algorithm files"""

    detection_data = DetectionData

    detection_data.markers = np.zeros((0, 6))

    for idx, file_name in file_generator:
        if idx % 100 == 0:
            logger.debug(f"Procesing frame {idx}")
        with open(file_name, 'r') as file:
            file_content = json.load(file)

            for person_num, person in enumerate(file_content):
                keypoints_in_frame = file_content[person]
                np_keypoints = np.array(keypoints_in_frame).reshape((-1, 2))
                # print(np_keypoints)
                for i, (u, v) in enumerate(np_keypoints):
                    detection_data.markers = np.vstack([detection_data.markers, [idx, person_num, i, u, v, 1.0]])

    detection_data.author = "A0_ArtTrack"
    detection_data.framesize = (-1, -1)

    return detection_data


def parse_a1(file_generator, bodyparts):
    detection_data = DetectionData
    detection_data.markers = np.zeros((0, 6))

    for idx, file_name in file_generator:
        if idx % 100 == 0:
            logger.debug(f"Procesing frame {idx}")
        with open(file_name, 'r') as file:
            file_content = json.load(file)
            people = file_content["people"]

            for person_num, person in enumerate(people):
                keypoints_in_frame = person["pose_keypoints_2d"]
                np_keypoints = np.array(keypoints_in_frame).reshape((-1, 3))
                for i, (u, v, c) in enumerate(np_keypoints):
                    detection_data.markers = np.vstack([detection_data.markers, [idx, person_num, i, u, v, c]])

    if bodyparts == OpenPoseBody25:
        detection_data.author = "A1F0_openpose_body25"
    elif bodyparts == OpenPoseCoco:
        detection_data.author = "A1F1_openpose_coco"
    elif bodyparts == OpenPoseMpi:
        detection_data.author = "A1F2_openpose_mpi"
    else:
        detection_data.author = "Unknown"
        detection_data.framesize = (-1, -1)

    return detection_data


def parse_a2(file_generator, bodyparts):

    detection_data = DetectionData
    detection_data.markers = np.zeros((0, 6))

    for idx, file_name in file_generator:
        with open(file_name, 'r') as file:
            file_content = json.load(file)

            for frame_data in file_content.values():
                frame_pos = frame_data["framePos"]
                poses = frame_data["poses"]

                for person_num, pose in enumerate(poses):
                    keypoints = pose["keypoints"]
                    for i, keypoint in enumerate(keypoints):
                        u = keypoint["x"]
                        v = keypoint["y"]
                        c = keypoint["confidence"]
                        detection_data.markers = np.vstack([detection_data.markers, [frame_pos, person_num, i, u, v, c]])

    detection_data.author = "A2_posenet"
    return detection_data


def define_parser(fn, model):
    return {"parser_function": fn, "body_model": model}


algorithm_parser = {
    "A0_arttrack": define_parser(parse_a0, ArtTrack),
    "A1F0_openpose_body25": define_parser(parse_a1, OpenPoseBody25),
    "A1F1_openpose_coco": define_parser(parse_a1, OpenPoseCoco),
    "A1F2_openpose_mpi": define_parser(parse_a1, OpenPoseMpi),
    "A2_posenet": define_parser(parse_a2, PoseNet),
    "Annotation": define_parser(parse_annotation, Annotation)
}
