from os import path

import cv2 as cv
import numpy as np
import pandas as pd

from skimage.metrics._structural_similarity import structural_similarity as ssim
from skimage.metrics.simple_metrics import normalized_root_mse as mse
from skimage.metrics.simple_metrics import normalized_root_mse as rmse
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.utils.configuration import AlgorithmType
from toolkit.utils import file as file_utils
from toolkit.utils.timer import timed


def load_heatmap(heatmaps_grayscale_folder, dataset_name, view_type, algorithm_name):
    file_name = f"{dataset_name}_{view_type}_{algorithm_name}.png"
    file_path = path.join(heatmaps_grayscale_folder, file_name)
    return cv.imread(file_path, cv.IMREAD_GRAYSCALE)

interp = lambda image, range : np.interp(image, [0, max(1, np.max(image))], range)
normalized = lambda image: interp(image, [0, 1])

def preprocess(image, num_bins = 8):
    binned = lambda image, num_bins: np.digitize(image, np.linspace(0, max(1, np.max(image)), num_bins+1))
    return normalized(binned(normalized(image), num_bins))

def snr(f, f_hat):
    assert f.ndim == 2 and f_hat.ndim == 2, "Only single channel images are allowed!"
    assert f.shape == f_hat.shape, "Dimension mismatch in snr"

    return np.sum(f**2) / np.sum(((f - f_hat)**2))

def calculate_metrics(I_true, J_det):
    assert I_true.size == J_det.size, "Image dimensions are not equal"
    return {
        "RMSE": mse(I_true, J_det) ** (1 / 2),
        "NRMSE": rmse(I_true, J_det),
        "SSIM": ssim(I_true, J_det, gaussian_weights=True, sigma=1.5, use_sample_covariance=False),
        "PSNR": psnr(I_true, J_det),
        "SNR": snr(I_true, J_det)
    }

@timed
def generate_heatmaps_similarity(app_settings:ApplicationSettings, datasets):

    output_file = file_utils.join_folders([app_settings.output_folder(), "Tab7_Heatmap_results.csv"])

    if file_utils.exists(output_file):
        logger.debug(f"Skipping {output_file}. File already exists")
        return

    column_names = ["video", "algorithm", "bins",
                    "rmse_td", "nrmse_td", "ssim_td", "psnr_td",
                    "rmse_cam", "nrmse_cam", "ssim_cam", "psnr_cam"]

    full_result = pd.DataFrame(columns=column_names)
    new_row = {}

    gray_heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "gray_heatmaps"])

    for dataset in datasets:
        video_name = dataset.dataset_name
        new_row.update({"video": video_name})

        gt_td_heatmap = load_heatmap(gray_heatmaps_folder, video_name, "topdown", "annotation")
        gt_cam_heatmap = load_heatmap(gray_heatmaps_folder, video_name, "camera", "annotation")

        for algorithm_type in AlgorithmType:
            if algorithm_type == AlgorithmType.UNKNOWN:
                continue
            new_row.update({"algorithm": algorithm_type.name})

            dt_td_heatmap = load_heatmap(gray_heatmaps_folder, video_name, "topdown", algorithm_type)
            dt_cam_heatmap = load_heatmap(gray_heatmaps_folder, video_name, "camera", algorithm_type)

            for bins in [2, 4, 8, 16, 32, 64, 128, 256]:
                new_row.update({"bins": bins})

                gt_td_binned_heatmap = preprocess(gt_td_heatmap, bins)
                gt_cam_binned_heatmap = preprocess(gt_cam_heatmap, bins)
                dt_td_binned_heatmap = preprocess(dt_td_heatmap, bins)
                dt_cam_binned_heatmap = preprocess(dt_cam_heatmap, bins)

                td_results = calculate_metrics(gt_td_binned_heatmap, dt_td_binned_heatmap)
                cam_results = calculate_metrics(gt_cam_binned_heatmap, dt_cam_binned_heatmap)

                new_row.update({"rmse_td": td_results["RMSE"]})
                new_row.update({"nrmse_td": td_results["NRMSE"]})
                new_row.update({"ssim_td": td_results["SSIM"]})
                new_row.update({"psnr_td": td_results["PSNR"]})
                new_row.update({"rmse_cam": cam_results["RMSE"]})
                new_row.update({"nrmse_cam": cam_results["NRMSE"]})
                new_row.update({"ssim_cam": cam_results["SSIM"]})
                new_row.update({"psnr_cam": cam_results["PSNR"]})
                logger.debug(new_row)
                full_result = full_result.append(new_row, ignore_index=True)

    full_result.to_csv(output_file)