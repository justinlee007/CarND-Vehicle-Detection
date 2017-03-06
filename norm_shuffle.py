import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from trainer import *

if __name__ == "__main__":
    cars, notcars = load_features()

    car_features = extract_features(cars, color_space="RGB", spatial_size=(32, 32), hist_bins=32, orient=9,
                                    pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True,
                                    hog_feat=False)
    notcar_features = extract_features(notcars, color_space="RGB", spatial_size=(32, 32), hist_bins=32, orient=9,
                                       pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
                                       hist_feat=True, hog_feat=False)

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
