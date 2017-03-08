import argparse
import os

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for visualizing HOG feature extraction")
    parser.add_argument("-show", action="store_true", help="Visualize HOG image")
    parser.add_argument("-save", action="store_true", help="Save HOG image")
    results = parser.parse_args()
    visualize = bool(results.show)
    save_examples = bool(results.save)

    # Read in the image
    image_file = "./test_images/31.png"
    image = mpimg.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = detection.get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True,
                                                     feature_vec=False)

    # Plot the examples
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Example Car Image", fontsize=24)
    ax2.imshow(hog_image, cmap="gray")
    ax2.set_title("HOG Visualization", fontsize=24)
    if visualize:
        plt.show()
    if save_examples:
        save_file_name = "hog_features_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")
