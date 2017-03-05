import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from detection_functions import *

if __name__ == "__main__":
    image = mpimg.imread("test_images/cutout1.jpg")
    rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, vis=True)

    # Plot a figure with all three bar charts
    if rh is not None:
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title("R Histogram")
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title("G Histogram")
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title("B Histogram")
        plt.show()
