# from os import path
#
# import cv2 as cv
# import numpy as np
#
# from skimage.metrics._structural_similarity import structural_similarity as ssim
# from skimage.metrics.simple_metrics import normalized_root_mse as mse
# from skimage.metrics.simple_metrics import normalized_root_mse as rmse
# from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr
#
# from toolkit.importer.argument_parser import ApplicationSettings
# from toolkit.logger import logger
# from toolkit.utils.configuration import AlgorithmType
# from toolkit.utils import file as file_utils
#
# def calculate_metrics(I_true, J_det):
#     assert I_true.size == J_det.size, "Image dimensions are not equal"
#     return {
#         "RMSE": mse(I_true, J_det) ** (1 / 2),
#         "NRMSE": rmse(I_true, J_det),
#         "SSIM": ssim(I_true, J_det, gaussian_weights=True, sigma=1.5, use_sample_covariance=False),
#         "PSNR": psnr(I_true, J_det),
#         "SNR": snr(I_true, J_det)
#     }
#
# def snr(f, f_hat):
#     assert f.ndim == 2 and f_hat.ndim == 2, "Only single channel images are allowed!"
#     assert f.shape == f_hat.shape, "Dimension mismatch in snr"
#
#     return np.sum(f**2) / np.sum(((f - f_hat)**2))
#
# interp = lambda image, range : np.interp(image, [0, max(1, np.max(image))], range)
# normalized = lambda image: interp(image, [0, 1])
#
# def preprocess(image, num_bins = 8):
#     binned = lambda image, num_bins: np.digitize(image, np.linspace(0, max(1, np.max(image)), num_bins+1))
#     return normalized(binned(normalized(image), num_bins))
#
# def compare_heatmaps_similarity(app_settings:ApplicationSettings, datasets):
#
#     output_file = file_utils.join_folders([app_settings.output_folder(), "Tab6_average_precisions.csv"])
#     if file_utils.exists(output_file):
#         logger.debug(f"Skipping {output_file}. File already exists")
#         return
#
#     gray_heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "gray_heatmaps"])
#
#     def load_heatmap(heatmaps_grayscale_folder, dataset_name, view_type, algorithm_name):
#         file_name = f"{dataset_name}_{view_type}_{algorithm_name}.png"
#         file_path = path.join(heatmaps_grayscale_folder, file_name)
#         return cv.imread(file_path, cv.IMREAD_GRAYSCALE)
#
#     for dataset in datasets:
#         dataset_name = dataset.dataset_name
#         logger.info(f"Processing {dataset_name}")
#
#         _, first_frame = cv.VideoCapture(dataset.video_file).read()
#
#         cv.namedWindow("GT", cv.WINDOW_KEEPRATIO)
#         cv.namedWindow("DT", cv.WINDOW_KEEPRATIO)
#         view_types = ["topdown", "camera"]
#         selected_view_types = view_types
#         for view_type in selected_view_types:
#             logger.info(f"{view_type=}:")
#             gt_heatmap = load_heatmap(gray_heatmaps_folder, dataset_name, view_type, "annotation")
#             gt_heatmap = preprocess(gt_heatmap)
#
#             # gt_overlayed = cv.addWeighted(gt_heatmap, 0.5, first_frame, 0.5, 0)
#             for algorithm_type in AlgorithmType:
#                 if algorithm_type == AlgorithmType.UNKNOWN:
#                     continue
#
#                 dt_heatmap = load_heatmap(gray_heatmaps_folder, dataset_name, view_type, algorithm_type)
#                 # for bins in [2, 4, 8, 16, 32, 64, 128, 256]:
#                 #     title = f"Bins_{bins}"
#                 #     cv.namedWindow(title, cv.WINDOW_KEEPRATIO)
#                 #     hm = preprocess(dt_heatmap, bins)
#                 #     hm = cv.applyColorMap((hm * 255).astype(np.uint8), cv.COLORMAP_TWILIGHT_SHIFTED)
#                 #
#                 #     cv.imshow(title, hm)
#                 # cv.waitKey(0)
#                 dt_heatmap = preprocess(dt_heatmap)
#
#                 metrics = calculate_metrics(gt_heatmap, dt_heatmap)
#
#                 new_row.update({"rmse": metrics["RMSE"], "nrmse": metrics["NRMSE"],
#                                 "ssim": metrics["ssim"], "psnr": metrics["psnr"]})
#                 print(f"{algorithm_type},{metrics['RMSE']},{metrics['NRMSE']},{metrics['SSIM']},{metrics['PSNR']}")
#
#                 if app_settings.render():
#                     gt_heatmap_color = cv.applyColorMap((gt_heatmap * 255).astype(np.uint8), cv.COLORMAP_TWILIGHT_SHIFTED)
#                     dt_heatmap_color = cv.applyColorMap((dt_heatmap * 255).astype(np.uint8), cv.COLORMAP_TWILIGHT_SHIFTED)
#                     final = cv.hconcat([gt_heatmap_color, dt_heatmap_color])
#                     cv.imshow("RENDER", final)
#                     cv.waitKey(1000)