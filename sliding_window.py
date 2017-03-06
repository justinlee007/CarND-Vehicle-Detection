import matplotlib

matplotlib.use("TkAgg")

from detection_functions import *

if __name__ == "__main__":
    image = mpimg.imread("test_images/bbox-example-image.jpg")
    y = image.shape[0]
    y_start = 400  # for size=64 and 128
    # y_start = 336  # for size = 256
    # y_stop = y - 128  # for size=64
    y_stop = y - 64  # for size=128
    # y_stop = y  # for size=256
    # size = 64
    size = 128
    # size = 256
    overlap = 0.5
    x = image.shape[1]
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start, y_stop], xy_window=(size, size),
                           xy_overlap=(overlap, overlap))

    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()
