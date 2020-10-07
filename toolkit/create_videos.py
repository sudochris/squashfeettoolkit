import os
import cv2 as cv
import numpy as np
from tqdm import tqdm
from toolkit.datamodel.bodyparts import Annotation, ArtTrack, OpenPoseBody25, OpenPoseCoco, OpenPoseMpi, PoseNet
from toolkit.datamodel.dataset import load_annotation_data, load_detection_data
from toolkit.heatmap import calibration, transformations
from toolkit.heatmap.accumulator import Accumulator
from toolkit.importer.argument_parser import StringOption, BooleanOption, application_settings_from_args
from toolkit.logger import logger
from toolkit.scripts.generate_heatmaps import AccumulatorType
from toolkit.utils.configuration import load_datasets, AlgorithmType
from toolkit.utils import marker as m_utils

def transform_accumulator_to_heatmap(accumulator):
    return transformations.logarithmic(accumulator)

def rect_text(img, text):
    (x_offset, y_offset) = (32, 32)
    cv.rectangle(img, (x_offset, y_offset), (200+x_offset, 64+y_offset), (255, 255, 255), cv.FILLED)
    cv.rectangle(img, (x_offset, y_offset), (200+x_offset, 64+y_offset), (0, 0, 0), 2)
    cv.putText(img, text, (20+x_offset, 54+y_offset), cv.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0), thickness=3, lineType=cv.LINE_AA)

def draw_border(img, border_size):
    cv.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (220, 220, 220), border_size)

def draw_court_lines_flip(img):
    h, w, _ = img.shape

    def w2i(x, z):
        # todo replace with np.interp
        return (int((x / 6.4) * w), int((z / 9.75) * h))

    cv.line(img, w2i(3.2, 0.00), w2i(3.2, 4.21), (0, 0, 255), 4, cv.LINE_AA)
    cv.line(img, w2i(0.0, 4.21), w2i(6.4, 4.21), (0, 0, 255), 4, cv.LINE_AA)

    cv.line(img, w2i(0.0, 2.61), w2i(1.6, 2.61), (0, 0, 255), 4, cv.LINE_AA)
    cv.line(img, w2i(1.6, 2.61), w2i(1.6, 4.21), (0, 0, 255), 4, cv.LINE_AA)
    cv.line(img, w2i(4.8, 4.21), w2i(4.8, 2.61), (0, 0, 255), 4, cv.LINE_AA)
    cv.line(img, w2i(4.8, 2.61), w2i(6.4, 2.61), (0, 0, 255), 4, cv.LINE_AA)

    return np.flipud(img)

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
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        capture = cv.VideoCapture(dataset.video_file)

        annotation_data = load_annotation_data(dataset)
        num_frames = max(annotation_data.markers[:, 0])
        (I_COLS, I_ROWS) = annotation_data.framesize

        camera_accumulators = {
            AccumulatorType.ANNOTATION: Accumulator(I_ROWS, I_COLS),
            AccumulatorType.DETECTIONS: {}
        }

        top_down_accumulators = {
            AccumulatorType.ANNOTATION: Accumulator(975, 640),
            AccumulatorType.DETECTIONS: {}
        }

        camera_accumulator_points = {
            AccumulatorType.ANNOTATION: [],
            AccumulatorType.DETECTIONS: {
                AlgorithmType.A0_ARTTRACK: [],
                AlgorithmType.A1F0_OPENPOSE_BODY25: [],
                AlgorithmType.A1F1_OPENPOSE_COCO: [],
                AlgorithmType.A1F2_OPENPOSE_MPI: [],
                AlgorithmType.A2_POSENET: []
            }
        }

        top_down_accumulator_points = {
            AccumulatorType.ANNOTATION: [],
            AccumulatorType.DETECTIONS: {
                AlgorithmType.A0_ARTTRACK: [],
                AlgorithmType.A1F0_OPENPOSE_BODY25: [],
                AlgorithmType.A1F1_OPENPOSE_COCO: [],
                AlgorithmType.A1F2_OPENPOSE_MPI: [],
                AlgorithmType.A2_POSENET: []
            }
        }

        annotation_mids = [Annotation.LFootP1, Annotation.LFootP2, Annotation.RFootP1, Annotation.RFootP2]

        mids = {
            AlgorithmType.A0_ARTTRACK: [ArtTrack.RAnkle, ArtTrack.LAnkle],
            AlgorithmType.A1F0_OPENPOSE_BODY25: [OpenPoseBody25.RHeel, OpenPoseBody25.LHeel],
            AlgorithmType.A1F1_OPENPOSE_COCO: [OpenPoseCoco.RAnkle, OpenPoseCoco.LAnkle],
            AlgorithmType.A1F2_OPENPOSE_MPI: [OpenPoseMpi.RAnkle, OpenPoseMpi.LAnkle],
            AlgorithmType.A2_POSENET: [PoseNet.RAnkle, PoseNet.LAnkle],
        }

        calibration_data = calibration.load_from_file(dataset.calibration_file)

        logger.info(f"[{dataset_name}] Processing annotation data")

        detection_data = {}

        for algorithm in dataset.algorithms:
            algorithm_type = algorithm.algorithm_type
            camera_accumulators[AccumulatorType.DETECTIONS].update({algorithm_type: Accumulator(I_ROWS, I_COLS)})
            top_down_accumulators[AccumulatorType.DETECTIONS].update({algorithm_type: Accumulator(975, 640)})

            detection_data[algorithm] = load_detection_data(dataset, algorithm)  # Load detection data


        writer = cv.VideoWriter()
        color_scale = None
        for frame_pos in tqdm(range(int(num_frames))):
            _, dataset_frame = capture.read()

            camera_accumulator_points[AccumulatorType.ANNOTATION].clear()
            top_down_accumulator_points[AccumulatorType.ANNOTATION].clear()

            a_markers_in_frame = m_utils.filter_markers_by_frame(annotation_data.markers, frame_pos)
            a_markers_by_location = m_utils.filter_markers_by_location(a_markers_in_frame)
            a_markers_filtered = a_markers_by_location

            for a_marker in a_markers_filtered:
                f_pos, a_pid, a_mid, a_uv, a_c, a_uid = m_utils.unpack_marker(a_marker)
                if a_mid in annotation_mids:
                    a_px = a_uv * (I_COLS, I_ROWS)
                    camera_accumulators[AccumulatorType.ANNOTATION].add_point(int(a_px[1]), int(a_px[0]))

                    a_wp = calibration.estimate_from_pixel(calibration_data, [a_px[0] + 200, a_px[1]])
                    x_hm = int(np.interp(a_wp[0], [0, 6.4], [0, 640], 0, 640))
                    z_hm = int(np.interp(a_wp[2], [0, 9.75], [0, 975], 0, 975))
                    top_down_accumulators[AccumulatorType.ANNOTATION].add_point(z_hm, x_hm)

                    camera_accumulator_points[AccumulatorType.ANNOTATION].append(a_px)
                    top_down_accumulator_points[AccumulatorType.ANNOTATION].append((x_hm, z_hm))

            for algorithm in dataset.algorithms:
                algorithm_type = algorithm.algorithm_type
                d_markers_in_frame = m_utils.filter_markers_by_frame(detection_data[algorithm].markers, frame_pos)
                d_markers_by_location = m_utils.filter_markers_by_location(d_markers_in_frame)
                d_markers_filtered = d_markers_by_location
                for d_marker in d_markers_filtered:
                    f_pos, d_pid, d_mid, d_uv, d_c, d_uid = m_utils.unpack_marker(d_marker)
                    if d_mid in mids[algorithm_type]:
                        d_px = d_uv * (I_COLS, I_ROWS)
                        camera_accumulators[AccumulatorType.DETECTIONS][algorithm_type].add_point(int(d_px[1]), int(d_px[0]))

                        d_wp = calibration.estimate_from_pixel(calibration_data, [d_px[0] + 200, d_px[1]])
                        x_hm = int(np.interp(d_wp[0], [0, 6.4], [0, 640], 0, 640))
                        z_hm = int(np.interp(d_wp[2], [0, 9.75], [0, 975], 0, 975))
                        top_down_accumulators[AccumulatorType.DETECTIONS][algorithm_type].add_point(z_hm, x_hm)

            def apply_trafo_color_weight(accumulator, colormap = cv.COLORMAP_MAGMA):
                return cv.addWeighted(
                    cv.applyColorMap(transform_accumulator_to_heatmap(accumulator), colormap), 1.0,
                    dataset_frame, 0.5,0)

            def apply_trafo_color(accumulator, colormap = cv.COLORMAP_MAGMA):
                return cv.applyColorMap(transform_accumulator_to_heatmap(accumulator), colormap)

            annotation_cam = apply_trafo_color_weight(camera_accumulators[AccumulatorType.ANNOTATION])
            a0_cam = apply_trafo_color_weight(camera_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A0_ARTTRACK])
            a1f0_cam = apply_trafo_color_weight(camera_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A1F0_OPENPOSE_BODY25])
            a1f1_cam = apply_trafo_color_weight(camera_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A1F1_OPENPOSE_COCO])
            a1f2_cam = apply_trafo_color_weight(camera_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A1F2_OPENPOSE_MPI])
            a2_cam = apply_trafo_color_weight(camera_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A2_POSENET])

            annotation_td = apply_trafo_color(top_down_accumulators[AccumulatorType.ANNOTATION])
            a0_td = apply_trafo_color(top_down_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A0_ARTTRACK])
            a1f0_td = apply_trafo_color(top_down_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A1F0_OPENPOSE_BODY25])
            a1f1_td = apply_trafo_color(top_down_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A1F1_OPENPOSE_COCO])
            a1f2_td = apply_trafo_color(top_down_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A1F2_OPENPOSE_MPI])
            a2_td = apply_trafo_color(top_down_accumulators[AccumulatorType.DETECTIONS][AlgorithmType.A2_POSENET])

            h, w, _ = annotation_cam.shape
            td_h, td_w, _ = annotation_td.shape
            h_scl = h / td_h
            new_size = (int(td_w * h_scl), int(td_h * h_scl))

            annotation_td = cv.resize(annotation_td, new_size)
            a0_td = cv.resize(a0_td, new_size)
            a1f0_td = cv.resize(a1f0_td, new_size)
            a1f1_td = cv.resize(a1f1_td, new_size)
            a1f2_td = cv.resize(a1f2_td, new_size)
            a2_td = cv.resize(a2_td, new_size)

            rect_text(annotation_cam, "GT")
            rect_text(a0_cam, "A0")
            rect_text(a1f0_cam, "A1F0")
            rect_text(a1f1_cam, "A1F1")
            rect_text(a1f2_cam, "A1F2")
            rect_text(a2_cam, "A2")

            annotation_td= draw_court_lines_flip(annotation_td)
            a0_td= draw_court_lines_flip(a0_td)
            a1f0_td= draw_court_lines_flip(a1f0_td)
            a1f1_td= draw_court_lines_flip(a1f1_td)
            a1f2_td= draw_court_lines_flip(a1f2_td)
            a2_td= draw_court_lines_flip(a2_td)

            annotation_final = cv.hconcat([annotation_cam, annotation_td])

            a0_final = cv.hconcat([a0_cam, a0_td])
            a1f0_final = cv.hconcat([a1f0_cam, a1f0_td])
            a1f1_final = cv.hconcat([a1f1_cam, a1f1_td])
            a1f2_final = cv.hconcat([a1f2_cam, a1f2_td])
            a2_final = cv.hconcat([a2_cam, a2_td])

            #for cam_pt, (td_x, td_y) in zip(camera_accumulator_points[AccumulatorType.ANNOTATION], top_down_accumulator_points[AccumulatorType.ANNOTATION]):
            #     start_line = tuple(cam_pt.astype(np.int))
            #     end_line = (td_x+w, h-td_y)
            #     cv.line(annotation_final, start_line, end_line, (255, 0, 255), 2, cv.LINE_AA)

            border_size = 8
            draw_border(annotation_final, border_size)
            draw_border(a0_final, border_size)
            draw_border(a1f0_final, border_size)
            draw_border(a1f1_final, border_size)
            draw_border(a1f2_final, border_size)
            draw_border(a2_final, border_size)

            # Inefficient, but its okay since its only run for creating the images
            _, width, _ = a2_final.shape
            legend_height = 320
            legend = np.zeros((legend_height, width*2, 3), dtype=np.uint8)

            if color_scale is None:
                lh, lw, _ = legend[16:legend_height//2, width:].shape
                color_scale = np.meshgrid(np.linspace(0, 1, lw), np.linspace(0, 1, lh))[0] * 255

            legend[16:legend_height // 2, width:] = cv.applyColorMap(color_scale.astype(np.uint8), cv.COLORMAP_MAGMA)
            cv.rectangle(legend, (width, 0), (width*2, legend_height//2), (255, 255, 255), 4, cv.LINE_AA)

            cv.putText(legend, f"Frame: {frame_pos} / {int(num_frames)}", (0, legend_height//2), cv.FONT_HERSHEY_DUPLEX, 2.0, (255, 255 ,255), thickness=3, lineType=cv.LINE_AA)
            cv.putText(legend, "0 [low]", (width, legend_height - legend_height // 4), cv.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), thickness=3, lineType=cv.LINE_AA)
            cv.putText(legend, "[high] 1", (width*2 - 280, legend_height - legend_height//4), cv.FONT_HERSHEY_DUPLEX, 2.0, (255, 255 ,255), thickness=3, lineType=cv.LINE_AA)

            result = cv.vconcat([
                cv.hconcat([annotation_final,   a1f0_final]),
                cv.hconcat([a0_final,           a1f1_final]),
                cv.hconcat([a2_final,           a1f2_final]),
                legend
            ])

            result = cv.copyMakeBorder(result, 16 ,16 ,16, 16, cv.BORDER_CONSTANT, value=(255, 255, 255))

#            final_path = os.path.join(dataset_output_path, "frame_{:0>8}.png")
#            cv.imwrite(final_path.format(frame_pos), result)

            if not writer.isOpened():
                result_height, result_width, _ = result.shape
                logger.info("Open VideoFile")
                output_file = os.path.join(app_settings.output_folder(), f"{dataset_name}.mp4")
                writer.open(output_file, cv.VideoWriter_fourcc(*"mp4v"), 1, (result_width, result_height))
            writer.write(result)

            if app_settings.render():
                cv.imshow("RENDER", result)
                cv.waitKey(1)

        writer.release()