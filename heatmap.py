import argparse
import glob

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import trainer
import detection
import scipy.ndimage.measurements as measurement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for visualizing color histogram feature extraction")
    parser.add_argument("-show", action="store_true", help="Show each channel histogram")
    parser.add_argument("-save", action="store_true", help="Save histogram visualization")
    results = parser.parse_args()
    show = bool(results.show)
    save = bool(results.save)

    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = trainer.load_svc()
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

        img_boxes, heatmap = detection.find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                                 cell_per_block, spatial_size, hist_bins)

        heatmap = detection.apply_threshold(heatmap, 2)

        # Find final boxes from heatmap using label function
        labels = measurement.label(heatmap)

        draw_img = detection.draw_labeled_bboxes(np.copy(image), labels)
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title("Car Positions")
        plt.subplot(122)
        plt.imshow(heatmap, cmap="hot")
        plt.title("Heat Map")
        plt.show()
