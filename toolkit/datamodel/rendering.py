import cv2 as cv
import numpy as np
from enum import IntEnum

def construct_parameter_object(name, value, max_value):
    return {"name": name, "value": value, "max_value": max_value}

class Rendering:

    class MarkerType(IntEnum):
        ANNOTATION = cv.MARKER_CROSS,
        DETECTION = cv.MARKER_TILTED_CROSS

    class Action(IntEnum):
        NOTHING = 0,
        EXIT = 1

    def __init__(self, is_enabled, default_waitkey = 0) -> None:
        super().__init__()
        self.is_enabled = is_enabled
        self.none_fn = lambda _: None
        self.window_name = "Result"
        self.overlay_texts = []

        self.parameter = [
            construct_parameter_object("alpha", 50, 100),
            construct_parameter_object("wait_key", default_waitkey, 1000)
        ]

        self.setup_rendering()

    def reset_video_file(self):
        self.set_video_frame_pos(0)

    def set_video_frame_pos(self, frame_pos):
        self.capture.set(cv.CAP_PROP_POS_FRAMES, frame_pos)

    def setup_rendering(self):
        if self.is_enabled:
            cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
            for parameter in self.parameter:
                cv.createTrackbar(parameter["name"], self.window_name, parameter["value"], parameter["max_value"], self.none_fn)

    def set_video_file(self, video_file):
        self.capture = cv.VideoCapture(video_file)
        width = self.capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.overlay = np.zeros(shape=(int(height), int(width), 3))
        self.reset_video_file()

    def read_next_frame(self):
        if self.is_enabled:
            _, self.I = self.capture.read()
            self.overlay.fill(0)
            self.overlay_texts = []

    def draw_marker(self, image, image_pos, marker_type : MarkerType, color):
        if self.is_enabled:
            cv.drawMarker(image, tuple(image_pos), color, markerType=marker_type, markerSize=16, thickness=4)

    def draw_annotation_marker(self, image_pos, color=(0, 255, 0), marker_type = MarkerType.ANNOTATION):
        self.draw_marker(self.overlay, tuple(image_pos), marker_type, color)

    def draw_detection_marker(self, image_pos, color=(0, 0, 255), marker_type = MarkerType.DETECTION):
        self.draw_marker(self.overlay, tuple(image_pos), marker_type, color)

    def draw_line(self, start, end, color=(0, 0, 255)):
        cv.line(self.overlay, tuple(start), tuple(end), color, 2)

    def draw_text(self, position, text, color=(0, 255, 0)):
        cv.putText(self.overlay, text, tuple(position), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)

    def add_overlay_text(self, text="{:-<32}".format(""), color=(0, 0, 255)):
        self.overlay_texts.append({"text": text, "color" : color})

    def render(self, with_waitkey = True):
        if self.is_enabled:

            y_start = 20
            spacing = 30
            for i, overlay_object in enumerate(self.overlay_texts):
                overlay_text = overlay_object["text"]
                overlay_text_color = overlay_object["color"]
                self.draw_text((20, y_start + (i * spacing)), overlay_text, overlay_text_color)

            alpha = cv.getTrackbarPos("alpha", "Result") / 100.
            wait_key = cv.getTrackbarPos("wait_key", "Result")

            R = cv.addWeighted(self.I, alpha, self.overlay, 1.0, 0.0, dtype=cv.CV_8UC3)
            cv.imshow(self.window_name, R)
            if with_waitkey:
                key = cv.waitKey(wait_key)
                if key == ord('q'):
                    return Rendering.Action.EXIT

        return Rendering.Action.NOTHING