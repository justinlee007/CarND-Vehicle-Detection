import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import glob
from detection_functions import *

if __name__ == "__main__":
    not_car_images = glob.glob("non-vehicles/*/*.png")
    car_images = glob.glob("vehicles/*/*.png")
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
    plt.imshow(image, cmap="gray")
    plt.title("Example Car Image")
    plt.subplot(122)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG Visualization")
    plt.show()
