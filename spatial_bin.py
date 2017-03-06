import matplotlib

matplotlib.use("TkAgg")

from detection_functions import *

if __name__ == "__main__":
    images = glob.glob("test_images/cutout*.jpg")

    for file_name in images:
        # Read in an image
        image = mpimg.imread(file_name)
        feature_vec = bin_spatial(image, size=(32, 32))

        # Plot features
        plt.plot(feature_vec)
        plt.title("Spatially Binned Features")
        plt.show()
