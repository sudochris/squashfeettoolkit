from os import path

import cv2 as cv
import numpy as np

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.utils.configuration import AlgorithmType
from toolkit.utils import file as file_utils
from toolkit.utils.timer import timed


def get_colorize_heatmaps_function(with_postprocessing: bool):
    @timed
    def colorize_heatmaps(app_settings:ApplicationSettings, datasets):

        gray_heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "gray_heatmaps"])
        color_heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "color_heatmaps"])
        file_utils.make_dirs(color_heatmaps_folder)

        def load_heatmap(heatmaps_grayscale_folder, dataset_name, view_type, algorithm_name):
            file_name = "{}_{}_{}{}.png".format(dataset_name, view_type, algorithm_name,
                                                "_FILTERED" if with_postprocessing else "")
            file_path = path.join(heatmaps_grayscale_folder, file_name)
            return cv.imread(file_path, cv.IMREAD_GRAYSCALE)

        for dataset in datasets:
            dataset_name = dataset.dataset_name
            logger.info(f"Colorizing {dataset_name}")

            print(dataset.court_image_file)
            capture = cv.VideoCapture(dataset.video_file)
            capture.set(cv.CAP_PROP_POS_FRAMES, capture.get(cv.CAP_PROP_FRAME_COUNT)-100)


            gt_cam_heatmap = load_heatmap(gray_heatmaps_folder, dataset_name, "camera", "annotation")
            gt_td_heatmap = load_heatmap(gray_heatmaps_folder, dataset_name, "topdown", "annotation")

            gt_cam_heatmap_color = cv.applyColorMap(gt_cam_heatmap, cv.COLORMAP_MAGMA)
            gt_td_heatmap_color = cv.applyColorMap(gt_td_heatmap, cv.COLORMAP_MAGMA)

            dataset_frame = cv.imread(dataset.court_image_file)

            gt_cam_overlayed = cv.addWeighted(gt_cam_heatmap_color, 1.0, dataset_frame, .5, 0)

            def get_overlayed_heatmap_for(algorithm_type):
                dt_heatmap = load_heatmap(gray_heatmaps_folder, dataset_name, "camera", algorithm_type)
                dt_heatmap_color = cv.applyColorMap(dt_heatmap, cv.COLORMAP_MAGMA)
                return cv.addWeighted(dt_heatmap_color, 1.0, dataset_frame, 0.5, 0)

            def get_td_heatmap_for(algorithm_type):
                td_heatmap = load_heatmap(gray_heatmaps_folder, dataset_name, "topdown", algorithm_type)
                return np.flipud(cv.applyColorMap(td_heatmap, cv.COLORMAP_MAGMA))

            a0 = get_overlayed_heatmap_for(AlgorithmType.A0_ARTTRACK)
            a1f0 = get_overlayed_heatmap_for(AlgorithmType.A1F0_OPENPOSE_BODY25)
            a1f1 = get_overlayed_heatmap_for(AlgorithmType.A1F1_OPENPOSE_COCO)
            a1f2 = get_overlayed_heatmap_for(AlgorithmType.A1F2_OPENPOSE_MPI)
            a2 = get_overlayed_heatmap_for(AlgorithmType.A2_POSENET)
            h, w, _  = gt_cam_overlayed.shape
            td_h, td_w, _ = gt_td_heatmap_color.shape

            h_scl = h / td_h
            new_size = (int(td_w * h_scl), int(td_h * h_scl))

            gt_td_heatmap_color = cv.resize(np.flipud(gt_td_heatmap_color), new_size)
            a0_td = cv.resize(get_td_heatmap_for(AlgorithmType.A0_ARTTRACK), new_size)
            a1f0_td = cv.resize(get_td_heatmap_for(AlgorithmType.A1F0_OPENPOSE_BODY25), new_size)
            a1f1_td = cv.resize(get_td_heatmap_for(AlgorithmType.A1F1_OPENPOSE_COCO), new_size)
            a1f2_td = cv.resize(get_td_heatmap_for(AlgorithmType.A1F2_OPENPOSE_MPI), new_size)
            a2_td = cv.resize(get_td_heatmap_for(AlgorithmType.A2_POSENET), new_size)

            overlay_result = cv.vconcat([cv.hconcat([gt_cam_overlayed, gt_td_heatmap_color, a1f0, a1f0_td]),
                                         cv.hconcat([a0,               a0_td,               a1f1, a1f1_td]),
                                         cv.hconcat([a2,               a2_td,               a1f2, a1f2_td])])

            def overlay_lines(img):
                def w2i(wx, wy):
                    h, w, _ = img.shape
                    xRel = wx / 6.4
                    yRel = wy / 9.75

                    return (int(w * xRel), h - int(h * yRel))

                cv.line(img, w2i(3.2, 0.00), w2i(3.2, 4.26), (255, 255, 255), 2, cv.LINE_AA)
                cv.line(img, w2i(0.0, 4.26), w2i(6.4, 4.26), (255, 255, 255), 2, cv.LINE_AA)

                cv.line(img, w2i(0.0, 2.61), w2i(1.6, 2.61), (255, 255, 255), 2, cv.LINE_AA)
                cv.line(img, w2i(1.6, 2.61), w2i(1.6, 4.26), (255, 255, 255), 2, cv.LINE_AA)

                cv.line(img, w2i(6.4, 2.61), w2i(4.8, 2.61), (255, 255, 255), 2, cv.LINE_AA)
                cv.line(img, w2i(4.8, 2.61), w2i(4.8, 4.26), (255, 255, 255), 2, cv.LINE_AA)

            overlay_lines(gt_td_heatmap_color)
            overlay_lines(a0_td)
            overlay_lines(a1f0_td)
            overlay_lines(a1f1_td)
            overlay_lines(a1f2_td)
            overlay_lines(a2_td)

            image_map = {
                "_GT_CAM": gt_cam_overlayed, "_GT_TD": gt_td_heatmap_color,
                "_A0_CAM": a0, "_A0_TD": a0_td,
                "_A1F0_CAM": a1f0, "_A1F0_CAM_TD": a1f0_td,
                "_A1F1_CAM": a1f1, "_A1F1_CAM_TD": a1f1_td,
                "_A1F2_CAM": a1f2, "_A1F2_CAM_TD": a1f2_td,
                "_A2_CAM": a2, "_A2_TD": a2_td
            }

            for suffix, image in image_map.items():
                overlayed_heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "overlayed_heatmaps"])
                file_utils.make_dirs(overlayed_heatmaps_folder)
                output_filename = "{}{}{}.png".format(dataset_name[:2], suffix, "_FILTERED" if with_postprocessing else "")
                full_output_file = file_utils.join_folders([overlayed_heatmaps_folder, output_filename])
                cv.imwrite(full_output_file, image)

            output_filename = "{}{}_colorized.png".format(dataset_name, "_FILTERED" if with_postprocessing else "")
            full_output_file = file_utils.join_folders([color_heatmaps_folder, output_filename])
            if not file_utils.exists(full_output_file):
                cv.imwrite(full_output_file, overlay_result)
                if app_settings.render():
                    cv.imshow("RENDER", overlay_result)
                    cv.waitKey(100)
            else:
                logger.debug(f"Skipping {full_output_file}. File already exists")
    return colorize_heatmaps