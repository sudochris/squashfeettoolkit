import cv2 as cv

from toolkit.importer.argument_parser import StringOption, BooleanOption, \
    application_settings_from_args

from toolkit.logger import logger
from toolkit.scripts.render_pose_figure import render_pose_figure
from toolkit.scripts.ap_results import ap_results
from toolkit.scripts.colorize_heatmaps import colorize_heatmaps
from toolkit.scripts.frame_type_significance import frame_type_significance
from toolkit.scripts.frameresult_metrics import frameresult_metrics
from toolkit.scripts.generate_heatmaps import generate_heatmaps
from toolkit.scripts.generate_heatmaps_similarity import generate_heatmaps_similarity
from toolkit.scripts.match_experiments import match_experiments
from toolkit.scripts.numpy_converter import numpy_converter
from toolkit.scripts.overall_statistics import overall_statistics
from toolkit.scripts.quantized_heatmaps import quantized_heatmaps

from toolkit.utils.configuration import load_datasets


if __name__ == '__main__':
    argument_options = [
        StringOption("description", "Dataset definition (default: dataset.json)", "dataset.json"),
        StringOption("output", "Output folder (default: output)", "output"),
        BooleanOption("debug", "Print Debug Output"),
        BooleanOption("render", "Enables rendering while processing (slow)")
    ]

    app_settings = application_settings_from_args(argument_options)

    selected_log_level = logger.LogLevel.DEBUG if app_settings.debug() else logger.LogLevel.INFO
    logger.select_log_level(selected_log_level)

    if app_settings.render():
        cv.namedWindow("RENDER", cv.WINDOW_KEEPRATIO)

    logger.info("Loading datasets")                 # 1. Load the datasets
    datasets = load_datasets(app_settings.description_file())

    def call_script(display_name, script_entry):
        logger.info(f"\t{display_name}")
        script_entry(app_settings, datasets)

    logger.info("Calling scripts")                  # 2. Call all scripts in correct order.

    call_script("numpy_converter", numpy_converter)                           # a) Numpy converter for easy numpy access
    call_script("overall_statistics", overall_statistics)                     # b) Dataset statistics
    call_script("match_experiments", match_experiments)                       # c) All actual experiment scripts
    call_script("frameresult_metrics", frameresult_metrics)                   # d) Calculate
    call_script("frame_type_significance", frame_type_significance)           # e) Significance tests (FA, FR, FN)
    call_script("ap_results", ap_results)                                     # f) Average Precision results
    call_script("generate_heatmaps", generate_heatmaps)                       # g) Generate grayscale heatmaps
    call_script("colorize_heatmaps", colorize_heatmaps)                       # h) Colorize Heatmaps
    call_script("generate_heatmaps_similarity", generate_heatmaps_similarity) # i) Generate similarity
    call_script("quantized_heatmaps", quantized_heatmaps)                     # j) Quantized Heatmap

#    call_script("render_pose_figure", render_pose_figure)                     # k) Render COCO model for a single frame (Fig. 1)

    logger.info(f"Done. Check '{app_settings.output_folder()}' and results folders.")

    cv.destroyAllWindows()