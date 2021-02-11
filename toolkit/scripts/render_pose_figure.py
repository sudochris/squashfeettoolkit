import os

from typing import List
import numpy as np
import cv2 as cv

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.utils.configuration import Dataset
from toolkit.utils.timer import timed
import toolkit.utils.file as file_utils

@timed
def render_pose_figure(app_settings:ApplicationSettings, datasets: List[Dataset]):
    dataset = datasets[1]
    frame_pos = 25

    loaded_markers = np.load(dataset.algorithms[2].numpy_file)

    # Extract only relevant markers for the selected frame. Skip eyes and ears ( id >= 14)
    markers_in_frame = loaded_markers[(loaded_markers[:, 0] == frame_pos) &  (loaded_markers[:, 2] < 14)]

    p1_markers = markers_in_frame[markers_in_frame[:, 1] == 0]
    p2_markers = markers_in_frame[markers_in_frame[:, 1] == 1]


    in_filename = os.path.join(os.path.dirname(__file__), "render_pose_image.png")
    out_filename = file_utils.join_folders([app_settings.output_folder(), "render_pose_image_out.png"])

    I = cv.imread(in_filename)
    O = np.zeros_like(I)

    h, w = I.shape[:2]
    p1_markers = np.int32(p1_markers[:, -3:-1] * [w, h])
    p2_markers = np.int32(p2_markers[:, -3:-1] * [w, h])

    pose_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13)
    ]

    connections = {
        "RIGHT_ARM": ([1, 2, 3, 4], (100, 112, 36)),
        "LEFT_ARM": ([1, 5, 6, 7], (23, 198, 255)),
        "RIGHT_LEG": ([1, 8, 9, 10], (232, 137, 30)),
        "LEFT_LEG": ([1, 11, 12, 13], (103, 33, 224)),
        "NECK": ([0, 1], (26, 22, 254)),
    }

    CIRCLECOLOR = (128, 128, 128)


    for part, (connection_list, color) in connections.items():
        for idx in range(len(connection_list)-1):
            sp1, ep1 = p1_markers[connection_list[idx]], p1_markers[connection_list[idx+1]]
            sp2, ep2 = p2_markers[connection_list[idx]], p2_markers[connection_list[idx+1]]
            cv.line(O, tuple(sp1), tuple(ep1), color, 8, cv.LINE_AA)
            cv.line(O, tuple(sp2), tuple(ep2), color, 8, cv.LINE_AA)

    for p1_marker in p1_markers:
        cv.circle(O, tuple(p1_marker), 12, CIRCLECOLOR, 1, cv.LINE_AA)
    for p2_marker in p2_markers:
        cv.circle(O, tuple(p2_marker), 12, CIRCLECOLOR, 1, cv.LINE_AA)

    R = cv.addWeighted(I, 0.7, O, 1.0, 0)
    c0, r0, c1, r1 = [881, 624, 1327, 1038]
    R = R[r0:r1, c0:c1, :]
    cv.imwrite(out_filename, R)
