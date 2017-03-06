import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from trainer import *

cars, notcars = load_features()

# TODO play with these values to see how your classifier performs under different binning scenarios
spatial_size = 16
hist_bins = 16

car_features = extract_features(cars, color_space="RGB", spatial_size=(spatial_size, spatial_size), hist_bins=hist_bins,
                                spatial_feat=True, hist_feat=True, hog_feat=False)
notcar_features = extract_features(notcars, color_space="RGB", spatial_size=(spatial_size, spatial_size),
                                   hist_bins=hist_bins, spatial_feat=True, hist_feat=True, hog_feat=False)
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

print("Using spatial binning of: {} and {} histogram bins".format(spatial_size, hist_bins))
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
