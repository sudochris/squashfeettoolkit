from os import path

import cv2 as cv
import numpy as np

from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
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

@timed
def quantized_heatmaps(app_settings:ApplicationSettings, datasets):
    gray_heatmaps_folder = file_utils.join_folders([app_settings.output_folder(), "gray_heatmaps"])

    output_file = file_utils.join_folders([app_settings.output_folder(), "QuantizedHeatmaps.png"])
    if file_utils.exists(output_file):
        logger.debug(f"Skipping {output_file}. File already exists")
        return

    dataset = datasets[0]
    video_name = dataset.dataset_name

    gt_td_heatmap = load_heatmap(gray_heatmaps_folder, video_name, "topdown", "annotation")

    first_row = None
    second_row = None

    def append_to_row(row, img):
        if row is None:
            return img
        else:
            return cv.hconcat([row, img])

    for num, bins in enumerate([256, 128, 64, 32, 16, 8, 4, 2]):
        binned = preprocess(gt_td_heatmap, bins)
        binned_color = cv.applyColorMap((binned * 255).astype(np.uint8), cv.COLORMAP_MAGMA)
        binned_color = np.flipud(binned_color)
        s = 256
        rs = 550
        cs = 256
        binned_color = binned_color[rs:rs+s, cs:cs+s]

        if num < 4: # first row
            first_row = append_to_row(first_row, binned_color)
        else:
            second_row = append_to_row(second_row, binned_color)

    final_image = cv.vconcat([first_row, second_row])
    cv.imwrite(output_file, final_image)

    if app_settings.render():
        cv.imshow("RENDER", final_image)
        cv.waitKey(1000)