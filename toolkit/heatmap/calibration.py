import numpy as np
import cv2 as cv
from numpy.linalg import inv
from enum import IntEnum

class CalibKey(IntEnum):
    CAMERA_MATRIX = 0,
    ROTATION_MATRIX = 1,
    INV_CAMERA_MATRIX = 2,
    INV_ROTATION_MATRIX = 3,
    TRANSLATION_VECTOR = 4,
    DIST = 5


def load_from_file(calibration_file):
    loaded_data  = np.load(calibration_file)
    calibration_data = {CalibKey.CAMERA_MATRIX: loaded_data["camera_matrix"],
                        CalibKey.ROTATION_MATRIX: loaded_data["rotation_matrix"],
                        CalibKey.INV_CAMERA_MATRIX: loaded_data["inv_camera_matrix"],
                        CalibKey.INV_ROTATION_MATRIX: loaded_data["inv_rotation_matrix"],
                        CalibKey.TRANSLATION_VECTOR: loaded_data["translation_vector"],
                        CalibKey.DIST: loaded_data["dist"]}
    return calibration_data

def calibrate(frame, image_points, world_points):
    calibration_data = {}

    camera_matrix   = np.zeros((3, 3))
    rotation_matrix = np.zeros((3, 3))
    inv_camera_matrix   = np.zeros((3, 3))
    inv_rotation_matrix = np.zeros((3, 3))
    translation_vector = np.zeros((1, 3))
    dist = np.zeros((1, 5))

    w_points = np.array([world_points], np.float32)
    i_points = np.array([image_points], np.float32)

    frame_width, frame_height = frame.shape[0], frame.shape[1]
    image_size = (frame_height, frame_width)

    camera_matrix = np.array([
        [frame_width /2, 0,                200],
        [0,              frame_height / 2, 200],
        [0,              0,                  1]
    ])

    calibration_flags = cv.CALIB_USE_INTRINSIC_GUESS

    ret, camera_matrix, dist, _, _ = \
        cv.calibrateCamera(w_points, i_points, image_size,
                           camera_matrix, None,
                           flags=calibration_flags)

    ret, r_vecs_mat, translation_vector = cv.solvePnP(
        w_points, i_points, camera_matrix, dist
    )

    cv.Rodrigues(r_vecs_mat, rotation_matrix)

    inv_camera_matrix = inv(camera_matrix)
    inv_rotation_matrix = inv(rotation_matrix)

    calibration_data[CalibKey.CAMERA_MATRIX] = camera_matrix
    calibration_data[CalibKey.ROTATION_MATRIX] = rotation_matrix
    calibration_data[CalibKey.INV_CAMERA_MATRIX] = inv_camera_matrix
    calibration_data[CalibKey.INV_ROTATION_MATRIX] = inv_rotation_matrix
    calibration_data[CalibKey.TRANSLATION_VECTOR] = translation_vector
    calibration_data[CalibKey.DIST] = dist

    return calibration_data


def estimate_from_pixel(calibration_data, pixel_point, real_y=0):
    uv_point = np.array([pixel_point[0], pixel_point[1], 1])

    inv_rotation_matrix = calibration_data[CalibKey.INV_ROTATION_MATRIX]
    inv_camera_matrix = calibration_data[CalibKey.INV_CAMERA_MATRIX]
    translation_vector = calibration_data[CalibKey.TRANSLATION_VECTOR]

    tmp_mat0 = np.matmul(np.matmul(inv_rotation_matrix, inv_camera_matrix), uv_point)
    tmp_mat1 = np.matmul(inv_rotation_matrix, translation_vector)

    s = tmp_mat1[1][0] + real_y
    s /= tmp_mat0[1]

    tmp_mat2 = np.matmul(inv_rotation_matrix,
                            np.matmul(
                                s * inv_camera_matrix, uv_point
                            ) - translation_vector.T[0]
                         )

    return tmp_mat2

def project_to_image(calibration_data, points_3d):

    rotation_matrix = calibration_data[CalibKey.ROTATION_MATRIX]
    translation_vector = calibration_data[CalibKey.TRANSLATION_VECTOR]
    camera_matrix = calibration_data[CalibKey.CAMERA_MATRIX]
    dist = calibration_data[CalibKey.DIST]

    points_2d, _ = cv.projectPoints(
        np.array(points_3d, np.float32),
        rotation_matrix, translation_vector,
        camera_matrix, dist
    )

    return np.ravel(points_2d).reshape((-1,2))
