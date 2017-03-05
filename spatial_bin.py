import glob

import cv2
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Define a function to compute color histogram features
# Pass the color_space flag as 3-letter all caps string like 'HSV' or 'LUV' etc.
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features


if __name__ == '__main__':
    images = glob.glob("test_images/cutout*.jpg")

    for file_name in images:
        # Read in an image
        image = mpimg.imread(file_name)
        feature_vec = bin_spatial(image, color_space='RGB', size=(32, 32))

        # Plot features
        plt.plot(feature_vec)
        plt.title('Spatially Binned Features')
        plt.show()
