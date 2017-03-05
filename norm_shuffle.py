import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob

from lesson_functions import *


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, spatial_size=(32, 32), hist_bins=32):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        feature_image = mpimg.imread(file)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


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

    car_features = extract_features(cars, spatial_size=(32, 32), hist_bins=32)
    notcar_features = extract_features(notcars, spatial_size=(32, 32), hist_bins=32)

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        car_ind = np.random.randint(0, len(cars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title("Original Image (car)")
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title("Raw Features")
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title("Normalized Features")
        plt.show()
        notcar_ind = np.random.randint(0, len(notcars))
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(notcars[notcar_ind]))
        plt.title("Original Image (not car)")
        plt.subplot(132)
        plt.plot(X[notcar_ind])
        plt.title("Raw Features")
        plt.subplot(133)
        plt.plot(scaled_X[notcar_ind])
        plt.title("Normalized Features")
        plt.show()
