import matplotlib

matplotlib.use('TkAgg')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Returns HOG features and visualization
    :param img:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return: features, hog_image (if vis==True)
    """
    if vis == True:
        # Call with two outputs if vis==True
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    else:
        # Otherwise call with one output
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
                       feature_vector=feature_vec)
        return features


if __name__ == '__main__':
    not_car_images = glob.glob("non-vehicles_smallset/*/*.jpeg")
    car_images = glob.glob("vehicles_smallset/*/*.jpeg")
    cars = []
    notcars = []

    print("not_car_images size={}, car_images size={}".format(len(not_car_images), len(car_images)))
    for image in not_car_images:
        notcars.append(image)
    for image in car_images:
        cars.append(image)
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars))
    # Read in the image
    image = mpimg.imread(cars[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
