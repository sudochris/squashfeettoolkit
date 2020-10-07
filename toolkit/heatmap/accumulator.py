import time

import numpy as np
import cv2 as cv

import toolkit.heatmap.utils as utils
from toolkit.heatmap import transformations


class Accumulator(object):
    """
    An Accumulator is used as a representation for our heatmap
    """
    def __init__(self, rows: int, cols: int, custom_element = None) -> None:
        super().__init__()

        assert rows > 0, "Rows must be positive."
        assert cols > 0, "Cols must be positive."

        self.size = (rows, cols)
        (self.rows, self.cols) = self.size
        self.data = np.zeros(self.size)

        if custom_element is None:
            element_size = 21
            element_sigma = 5
            element_normalized = True

            self.element = utils.create_gauss(element_size, element_size,
                                              element_sigma, element_sigma,
                                              element_normalized)
        else:
            self.element = custom_element
            element_size = self.element.shape[0]

        self.element_half_size = element_size // 2

        self.data = np.pad(self.data, self.element_half_size+1,
                           constant_values=(1))

    def get(self):
        rc_start = self.element_half_size + 1
        rc_end = -self.element_half_size - 1
        return self.data[rc_start:rc_end, rc_start:rc_end]

    def add_point(self, r: int, c: int ):

        rcs = self.element_half_size

        row_start   = r - rcs + rcs
        row_end     = r + rcs + 1 + rcs
        col_start   = c - rcs + rcs
        col_end     = c + rcs + 1 + rcs
        self.data[row_start:row_end, col_start:col_end] += self.element

if __name__ == '__main__':
    def onmouse(event, x, y, flags, accumulator):
        if event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            accumulator.add_point(y, x)


    def custom_element(size, marker_type):
        my_element = np.zeros((size, size))
        cv.drawMarker(my_element, (size // 2, size // 2), (1), marker_type, size, line_type=cv.LINE_AA)
        return my_element

    (rows, cols) = (975, 640)

    element = custom_element(23, cv.MARKER_CROSS)
    accumulator = Accumulator(rows, cols, custom_element=None)

    cv.namedWindow("Heatmap_LIN", cv.WINDOW_KEEPRATIO)
    cv.namedWindow("Heatmap_LOG", cv.WINDOW_KEEPRATIO)
    cv.setMouseCallback("Heatmap_LIN", onmouse, accumulator)
    cv.setMouseCallback("Heatmap_LOG", onmouse, accumulator)
    running = True

    test_gauss = utils.create_gauss(rows, sigma_x=15)

    start = time.time()
    deltasum = 0

    from_world = lambda x, z: (int((x / 6.4) * cols), int((z / 9.75) * rows))

    lines = [
        {"from": from_world(3.20, 0.00), "to": from_world(3.20, 4.26)},
        {"from": from_world(0.00, 4.26), "to": from_world(6.40, 4.26)},
        {"from": from_world(0.00, 2.61), "to": from_world(1.60, 2.61)},
        {"from": from_world(4.80, 2.61), "to": from_world(6.40, 2.61)},
        {"from": from_world(1.60, 2.61), "to": from_world(1.60, 4.26)},
        {"from": from_world(4.80, 2.61), "to": from_world(4.80, 4.26)}
    ]

    parameters = {"fs": 0, "fe": 100, "ts": 0, "te": 100}

    def onchange(x):
        for parameter in parameters:
            parameters[parameter] = cv.getTrackbarPos(parameter, "Heatmap_LOG")

    for parameter in parameters:
        cv.createTrackbar(parameter, "Heatmap_LOG", parameters[parameter], 100, onchange)

    while running:
        delta = time.time() - start
        start = time.time()
        deltasum += delta * 20

        r = int((rows // 2) + (np.sin(deltasum) * 64))
        c = int((cols // 2) + (np.cos(deltasum) * 64))

        noise_r = np.random.randint(-4, 4)
        noise_c = np.random.randint(-4, 4)

        accumulator.add_point(r + noise_r, c + noise_c)

        heatmap_lin = transformations.linear(accumulator)
        heatmap_log = transformations.logarithmic(accumulator)


        heatmap_lin_color = cv.applyColorMap(heatmap_lin, cv.COLORMAP_TWILIGHT_SHIFTED)
        heatmap_log_color = cv.applyColorMap(heatmap_log, cv.COLORMAP_TWILIGHT_SHIFTED)

        for line in lines:
            cv.line(heatmap_lin_color, line["from"], line["to"], (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow("Heatmap_LIN", heatmap_lin_color)
        cv.imshow("Heatmap_LOG", heatmap_log_color)

        if cv.waitKey(1) == ord('q'):
            running = False
