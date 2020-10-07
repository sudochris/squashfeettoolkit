import numpy as np
import json

class DetectionData:

    def __init__(self) -> None:
        super().__init__()
        self.author = "Unknown"
        self.framesize = (-1, -1)
        self.markers = np.array([])

def append_index_column(the_markers):
    m_shape = the_markers.shape
    marker_indexe = np.arange(m_shape[0])

    result = np.zeros((m_shape[0], m_shape[1]+1))
    result[:, :-1] = the_markers
    result[:, -1] = marker_indexe

    return result
    pass

def load_annotation_data(dataset) -> DetectionData:
    result_data = DetectionData()

    with open(dataset.annotation_file, 'r') as file:
        json_data = json.load(file)
        annotation_info = json_data["annotationInfo"]

        result_data.author = annotation_info["author"]
        frame_width = annotation_info["framewidth"]
        frame_height = annotation_info["frameheight"]
        result_data.framesize = (frame_width, frame_height)

    loaded_markers = np.load(dataset.annotation_numpy)

    result_data.markers = append_index_column(loaded_markers)

    return result_data

def load_detection_data(dataset, algorithm) -> DetectionData:
    detection_data = DetectionData()
    detection_data.author = algorithm.algorithm_name
    detection_data.framesize = (-1, -1)

    loaded_markers = np.load(algorithm.numpy_file)
    detection_data.markers = append_index_column(loaded_markers)

    return detection_data

"""
markers is a numpy array with:
u, v => 0, 0 is top left
frame_pos, person_id, marker_id (see bodyparts), u (normalized x), v (normalized y), confidence
e.g.
0,  0,  0,  0.1,    0.3,    0.81
0,  0,  1,  0.5,    0.6,    0.76
0,  1,  2,  0.3,    0.6,    0.92
0,  1,  3,  0.6,    0.2,    0.12
1,  0,  0,  0.2,    0.5,    0.32
1,  0,  1,  0.4,    0.4,    0.81
1,  1,  2,  0.1,    0.3,    0.77
1,  1,  3,  0.2,    0.2,    0.65
...
"""
