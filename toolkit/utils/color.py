import cv2 as cv

from toolkit.datamodel.bodyparts import ArtTrack, OpenPoseBody25, OpenPoseCoco, OpenPoseMpi, PoseNet
from toolkit.utils.configuration import AlgorithmType

def color_marker_type(c, marker_type):
    return {"color": c, "marker_type": marker_type}

RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
PURPLE = (128, 0, 128)

color_definitions = {
    AlgorithmType.A0_ARTTRACK: {
        ArtTrack.LAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_DOWN),
        ArtTrack.RAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_UP)
    },
    AlgorithmType.A1F0_OPENPOSE_BODY25: {
        OpenPoseBody25.LHeel: color_marker_type(RED, cv.MARKER_TRIANGLE_DOWN),
        OpenPoseBody25.RHeel: color_marker_type(RED, cv.MARKER_TRIANGLE_UP)
    },
    AlgorithmType.A1F1_OPENPOSE_COCO: {
        OpenPoseCoco.LAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_DOWN),
        OpenPoseCoco.RAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_UP)
    },
    AlgorithmType.A1F2_OPENPOSE_MPI: {
        OpenPoseMpi.LAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_DOWN),
        OpenPoseMpi.RAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_UP)
    },
    AlgorithmType.A2_POSENET: {
        PoseNet.LAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_DOWN),
        PoseNet.RAnkle: color_marker_type(RED, cv.MARKER_TRIANGLE_UP)
    }

}

def algorithm_color_by_mid(d_mid, algorithm_type):
    if algorithm_type in color_definitions and d_mid in color_definitions[algorithm_type]:
        return color_definitions[algorithm_type][d_mid]["color"]
    return WHITE

def algorithm_marker_type_by_mid(d_mid, algorithm_type):
    if algorithm_type in color_definitions and d_mid in color_definitions[algorithm_type]:
        return color_definitions[algorithm_type][d_mid]["marker_type"]
    return cv.MARKER_SQUARE

def annotation_color_by_mid(a_mid):
    return YELLOW if a_mid % 2 == 0 else PURPLE

def annotation_marker_type_by_mid(a_mid):
    return cv.MARKER_TILTED_CROSS