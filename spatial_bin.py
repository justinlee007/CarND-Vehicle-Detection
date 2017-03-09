import argparse
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for visualizing spatial bin feature vector")
    parser.add_argument("-show", action="store_true", help="Show car image with spatially binned plot")
    parser.add_argument("-save", action="store_true", help="Save example spatial bin to disk")
    results = parser.parse_args()
    show = bool(results.show)
    save = bool(results.save)

    image_file = "test_images/cutout6.jpg"
    image = mpimg.imread(image_file)
    feature_vec = detection.bin_spatial(image, size=(32, 32))

    # Plot the examples
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(image)
    ax1.set_title("Original Car Image", fontsize=24)
    ax2.plot(feature_vec)
    ax2.set_title("Spatially Binned Features", fontsize=24)
    if show:
        plt.show()
    if save:
        save_file_name = "spatial_bin_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
        save_location = "./output_images/{}".format(save_file_name)
        f.savefig(save_location, bbox_inches="tight")
