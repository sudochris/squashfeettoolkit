import numpy as np
from numpy import linalg as linalg
from toolkit.utils import marker as m_utils


def classify_norm(d_markers, a_markers, thresh):
    # d_markers contains all class detections
    # a_markers contains all class labels

    classification_result = {
        "TP": [],
        "FP": []
    }
    matched_a_markers = []

    for d_marker in d_markers:
        _, _, d_mid, d_uv, d_c, d_uid = m_utils.unpack_marker(d_marker)

        d_is_tp = False
        best_distance = np.infty

        for a_marker in a_markers:
            if d_is_tp:
                continue

            _, _, a_mid, a_uv, a_c, a_uid = m_utils.unpack_marker(a_marker)
            if a_mid in matched_a_markers:
                continue

            distance = linalg.norm(a_uv - d_uv, 2)

            if distance < best_distance:
                best_distance = distance

            if distance <= thresh:
                matched_a_markers.append(a_uid) # a is "used"

                classification_result["TP"].append([d_uid, d_mid, d_uv, distance, True, d_c])
                d_is_tp = True
                break                           # Break, d_marker was matched

        if not d_is_tp: # d_marker is fp
            classification_result["FP"].append([d_uid, d_mid, d_uv, best_distance, False, d_c])

    return classification_result
