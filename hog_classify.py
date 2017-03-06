import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from detection_functions import *

if __name__ == "__main__":
    cars, notcars = load_features()

    ### TODO: Tweak these parameters and see how the results change.
    colorspace = "YUV"  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"

    t = time.time()
    car_features = extract_features(cars, color_space=colorspace, orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=False,
                                    hist_feat=False, hog_feat=True)
    notcar_features = extract_features(notcars, color_space=colorspace, orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=False,
                                       hist_feat=False, hog_feat=True)
    t2 = time.time()
    print(round(t2 - t, 2), "Seconds to extract HOG features...")
    # Create an array stack of feature vectors
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
    n_predict = 15
    print("My SVC predicts:\t\t{}".format(svc.predict(X_test[0:n_predict])))
    print("For these {} labels:\t{}".format(n_predict, y_test[0:n_predict]))
    t2 = time.time()
    print("Time to predict {} labels with SVC: {}s".format(n_predict, round(t2 - t, 5)))
