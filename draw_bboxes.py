import matplotlib

matplotlib.use("TkAgg")

from detection_functions import *

if __name__ == "__main__":
    image = mpimg.imread("test_images/bbox-example-image.jpg")
    # Add bounding boxes in this format, these are just example coordinates.
    bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]

    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.show()
