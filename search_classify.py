import matplotlib

matplotlib.use("TkAgg")
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from detection_functions import *

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    cars, notcars = load_features()
    ### TODO: Tweak these parameters and see how the results change.
    color_space = "YUV"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 16  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [400, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                    hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                       hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print("Using: {} orientations {} pixels per cell and {} cells per block".format(
        orient, pix_per_cell, cell_per_block))
    print("Feature vector length: {}".format(len(X_train[0])))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print("Time to train SVC:\t\t{:.2f}s".format((t2 - t)))
    # Check the score of the SVC
    print("Test Accuracy of SVC:\t{:.4f}".format(svc.score(X_test, y_test)))
    # Check the prediction time for a single sample
    t = time.time()

    image = mpimg.imread("test_images/bbox-example-image.jpg")
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32) / 255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=(128, 128),
                           xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size,
                                 hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()
