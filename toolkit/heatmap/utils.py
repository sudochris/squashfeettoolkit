import numpy as np
import cv2 as cv


def create_gauss(size_x: int = 21, size_y : int = None, sigma_x : int = 5, sigma_y: int = None, normalized = False):
    """Creates a 2d gaussian kernel with given sizes and variances"""
    if size_y is None:
        size_y = size_x

    if sigma_y is None:
        sigma_y = sigma_x

    assert isinstance(size_x, int), "size_x must be an integer!"
    assert isinstance(size_y, int), "size_y must be an integer!"
    assert size_x % 2 != 0, "size_x must be odd!"
    assert size_y % 2 != 0, "size_y must be odd!"

    gauss_x = cv.getGaussianKernel(size_x, sigma_x)
    gauss_y = cv.getGaussianKernel(size_y, sigma_y)

    gauss = gauss_x.T * gauss_y

    if normalized:
        gauss = np.interp(gauss, [0, np.max(gauss)], [0, 1])

    return gauss


if __name__ == '__main__':
    """Gauss Demo Application"""
    def min_max_val(min, max, val):
        return {"min": min, "max": max, "value": val}

    window_name = "Gauss"
    parameters = {"size_x": min_max_val(1, 512, 21),
                  "size_y": min_max_val(1, 512, 21),
                  "sigma_x": min_max_val(1, 53, 5),
                  "sigma_y": min_max_val(1, 53, 5)}

    do_nothing = lambda i: i
    cv.namedWindow(window_name, cv.WINDOW_KEEPRATIO)

    for parameter in parameters:
        cv.createTrackbar(parameter, window_name,
                          parameters[parameter]["value"],
                          parameters[parameter]["max"], do_nothing)

    running = True
    test_img = np.zeros((255, 255))
    while running:
        size_x = cv.getTrackbarPos("size_x", window_name)
        size_y = cv.getTrackbarPos("size_y", window_name)
        sigma_x = cv.getTrackbarPos("sigma_x", window_name)
        sigma_y = cv.getTrackbarPos("sigma_y", window_name)
        if size_x % 2 == 0:
            size_x += 1
        if size_y % 2 == 0:
            size_y += 1
        gauss_img = create_gauss(size_x, size_y, sigma_x, sigma_y, True)

        cv.imshow(window_name, gauss_img)

        key = cv.waitKey(1)
        if key == ord('q'):
            running = False