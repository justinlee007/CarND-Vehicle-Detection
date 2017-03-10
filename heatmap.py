import argparse
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import trainer
import detection
import scipy.ndimage.measurements as measurement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for visualizing vehicle detection heatmap")
    parser.add_argument("-show", action="store_true", help="Show vehicle detection boxes with corresponding heatmap")
    parser.add_argument("-save", action="store_true", help="Save example image to disk")
    parser.add_argument("-thresh", action="store_true", help="Apply threshold value of 2 for heatmap")
    parser.add_argument("-label", action="store_true", help="Use heatmap-based labeled bounding boxes")
    results = parser.parse_args()
    show = bool(results.show)
    save = bool(results.save)
    thresh = bool(results.thresh)
    label = bool(results.label)

    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = trainer.load_svc()
    image_file = "test_images/test6.jpg"
    image = mpimg.imread(image_file)
    y = image.shape[0]
    y_start = 400
    y_stop = y - 64
    scale = 1.5

    # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32) / 255
    print("min={}, max={}".format(np.min(image), np.max(image)))

    img_boxes, heatmap = detection.find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                             cell_per_block, spatial_size, hist_bins)

    if thresh:
        heatmap = detection.apply_threshold(heatmap, 2)

    if label:
        # Find final boxes from heatmap using label function
        labels = measurement.label(heatmap)
        draw_img = detection.draw_labeled_bboxes(np.copy(image), labels)
    else:
        draw_img = detection.draw_boxes(np.copy(image), img_boxes)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(draw_img)
    ax1.set_title("Car Positions", fontsize=24)
    ax2.imshow(heatmap, cmap="hot")
    ax2.set_title("Heat Map", fontsize=24)
    if show:
        plt.show()
    if save:
        save_file_name = "heatmap_{}{}{}".format("thresh_" if thresh else "", "label_" if label else "",
                                                  os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")
