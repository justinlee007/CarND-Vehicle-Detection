import glob
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import detection.norm_shuffle as feature

not_car_images = glob.glob("non-vehicles_smallset/*/*.jpeg")
car_images = glob.glob("vehicles_smallset/*/*.jpeg")
cars = []
notcars = []

print("not_car_images size={}, car_images size={}".format(len(not_car_images), len(car_images)))
for image in not_car_images:
    notcars.append(image)
for image in car_images:
    cars.append(image)

# TODO play with these values to see how your classifier
# performs under different binning scenarios
spatial = 40
histbin = 72

car_features = feature.extract_features(cars, cspace='RGB', spatial_size=(spatial, spatial), hist_bins=histbin,
                                        hist_range=(0, 256))
notcar_features = feature.extract_features(notcars, cspace='RGB', spatial_size=(spatial, spatial), hist_bins=histbin,
                                           hist_range=(0, 256))

# Create an array stack of feature vectors
X = np.vstack([car_features, notcar_features]).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

print("Using spatial binning of: {} and {} histogram bins".format(spatial, histbin))
print("Feature vector length:\t{}".format(len(X_train[0])))
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
