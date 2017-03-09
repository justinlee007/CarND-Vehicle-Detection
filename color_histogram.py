import argparse
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import detection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for visualizing color histogram feature extraction")
    parser.add_argument("-show", action="store_true", help="Show each channel histogram")
    parser.add_argument("-save", action="store_true", help="Save histogram visualization")
    results = parser.parse_args()
    show = bool(results.show)
    save = bool(results.save)

    image_file = "test_images/cutout2.jpg"
    image = mpimg.imread(image_file)
    rh, gh, bh, bincen, feature_vec = detection.color_hist(image, nbins=32, vis=True)

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(14, 3))
        plt.subplot(141)
        plt.title("Original Image")
        plt.imshow(image)
        plt.subplot(142)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 255)
        plt.title("R Histogram")
        plt.subplot(143)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 255)
        plt.title("G Histogram")
        plt.subplot(144)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 255)
        plt.title("B Histogram")
        if show:
            plt.show()
        if save:
            save_file_name = "color_histogram_{}".format(os.path.basename(image_file.replace(".jpg", ".png")))
            save_location = "./output_images/{}".format(save_file_name)
            fig.savefig(save_location, bbox_inches="tight")
