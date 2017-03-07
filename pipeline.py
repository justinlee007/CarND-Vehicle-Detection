import matplotlib

matplotlib.use("TkAgg")

from moviepy.editor import VideoFileClip
from trainer import *
from scipy.ndimage.measurements import label


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 1), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, labels, color=(0, 0, 1), thick=6):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))
        bboxes.append(bbox)
    # Draw the boxes on the image
    return draw_boxes(img, bboxes, color, thick)


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def process_image(image):
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = load_svc()

    y = image.shape[0]
    y_start = 400
    y_stop = y - 64
    scale = 1.5

    # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    draw_image = image.astype(np.float32) / 255
    # print("min={}, max={}".format(np.min(draw_image), np.max(draw_image)))

    img_boxes, heatmap = find_cars(draw_image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                   cell_per_block,
                                   spatial_size, hist_bins)

    heatmap = apply_threshold(heatmap, 2)

    # heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    image = draw_labeled_bboxes(image, labels, color=(0, 0, 255))
    return image


if __name__ == "__main__":
    output_file = "test.mp4"
    # output_file = "vehicle_detection.mp4"
    clip = VideoFileClip("test_video.mp4")
    # clip = VideoFileClip("project_video.mp4")
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_file, audio=False)
