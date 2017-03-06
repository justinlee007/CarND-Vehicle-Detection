import matplotlib

matplotlib.use("TkAgg")

from hog_subsample import *
from scipy.ndimage.measurements import label


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
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
    return draw_boxes(img, bboxes, color=(0, 0, 1), thick=6)


if __name__ == "__main__":
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = load_svc()
    images = glob.glob("test_images/test*.jpg")
    images.append("test_images/bbox-example-image.jpg")
    for image_file in images:
        image = mpimg.imread(image_file)
        y = image.shape[0]
        y_start = 400
        y_stop = y - 64
        scale = 1

        # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32) / 255
        print("min={}, max={}".format(np.min(image), np.max(image)))

        img_boxes, heatmap = find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                       cell_per_block, spatial_size, hist_bins)

        heatmap = apply_threshold(heatmap, 2)

        # heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        draw_img = draw_labeled_bboxes(np.copy(image), labels)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title("Car Positions")
        plt.subplot(122)
        plt.imshow(heatmap, cmap="hot")
        plt.title("Heat Map")
        plt.show()
