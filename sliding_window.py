import matplotlib

matplotlib.use("TkAgg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import draw_bboxes as boxes


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Takes an image, and returns a list of windows based on start and stop positions, window size and overlap fraction
    :param img:
    :param x_start_stop:
    :param y_start_stop:
    :param xy_window:
    :param xy_overlap:
    :return:
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice you"ll be considering windows one by one with your classifier
    # so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


if __name__ == "__main__":
    image = mpimg.imread("test_images/bbox-example-image.jpg")
    y = image.shape[0]
    # y_start = 400  # for size=64 and 128
    y_start = 336 # for size = 256
    # y_stop = y - 128  # for size=64
    # y_stop = y - 64  # for size=128
    y_stop = y  # for size=256
    # size = 64
    # size = 128
    size = 256
    overlap = 0.5
    x = image.shape[1]
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop], xy_window=(size, size),
                           xy_overlap=(overlap, overlap))

    window_img = boxes.draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()
