import argparse
import glob
import pickle

import matplotlib

matplotlib.use("TkAgg")
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from detection import *

from sklearn.model_selection import train_test_split


def train_classifier(color_space="RGB", orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL",
                     spatial_size=(32, 32), hist_bins=32, spatial_feat=True, hist_feat=True, hog_feat=True,
                     save_file=None):
    cars, notcars = load_features()
    print("orient={}, pix_per_cell={}, cell_per_block={} ".format(orient, pix_per_cell, cell_per_block))
    t = time.time()
    car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                    hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat,
                                       hog_feat=hog_feat)
    t2 = time.time()
    print("Feature extraction:\t\t{:.2f}s".format((t2 - t)))

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

    print("Feature vector length:\t{}".format(len(X_train[0])))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print("SVC training time:\t\t{:.2f}s".format((t2 - t)))
    # Check the score of the SVC
    print("Test accuracy:\t\t\t{:.4f}".format(svc.score(X_test, y_test)))
    if save_file is not None:
        print("Saving to file:\t\t\t{}".format(save_file))
        data = {"svc": svc, "scaler": X_scaler, "orient": orient, "pix_per_cell": pix_per_cell,
                "cell_per_block": cell_per_block, "spatial_size": spatial_size, "hist_bins": hist_bins}
        file = open(save_file, "wb")
        pickle.dump(data, file)
        file.close()


def load_features(use_smallset=False, use_subset=False, subset_size=1000):
    if use_smallset:
        not_car_images = glob.glob("non-vehicles_smallset/*/*.jpeg")
        car_images = glob.glob("vehicles_smallset/*/*.jpeg")
    else:
        not_car_images = glob.glob("non-vehicles/*/*.png")
        car_images = glob.glob("vehicles/*/*.png")
    cars = []
    notcars = []

    for image in car_images:
        cars.append(image)
    for image in not_car_images:
        notcars.append(image)

    if use_subset:
        if len(cars) > subset_size:
            random_indexes = np.random.randint(0, len(cars), subset_size)
            cars = np.array(cars)[random_indexes]
        if len(notcars) > subset_size:
            random_indexes = np.random.randint(0, len(notcars), subset_size)
            notcars = np.array(notcars)[random_indexes]
    print("len(cars)={}, len(notcars)={}".format(len(cars), len(notcars)))
    return cars, notcars


def load_svc(save_file="svc_pickle.p"):
    print("Loading svc from: {}".format(save_file))
    dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    return svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for training and storing SVC")
    parser.add_argument("-save", action="store_true", help="Pickle SVC, Scalar and training parameters")
    results = parser.parse_args()
    save_svc = bool(results.save)

    color_space = "YCrCb"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    save_file = "svc_pickle.p" if save_svc else None
    train_classifier(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                     spatial_feat, hist_feat, hog_feat, save_file=save_file)
