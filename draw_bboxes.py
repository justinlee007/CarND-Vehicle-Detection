import cv2
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Define a function that takes an image, a list of bounding boxes, and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


if __name__ == '__main__':
    image = mpimg.imread('test_images/bbox-example-image.jpg')
    # Add bounding boxes in this format, these are just example coordinates.
    bboxes = [((100, 100), (200, 200)), ((300, 300), (400, 400))]

    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.show()
