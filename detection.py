import time

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog


def convert_color(img, conv="RGB2YCrCb"):
    if conv == "RGB2YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == "BGR2YCrCb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == "RGB2LUV":
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis,
                       feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
def color_hist(img, nbins=32, vis=False):
    # Compute the histogram of the RGB channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Generating bin centers
    bin_edges = channel1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if vis:
        return channel1_hist, channel2_hist, channel3_hist, bin_centers, hist_features
    else:
        return hist_features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space="RGB", spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                     cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        image_features = extract_feature(image, color_space, spatial_size, hist_bins, orient, pix_per_cell,
                                         cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
        features.append(image_features)
    # Return list of feature vectors
    return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features() just for a single image rather than list of images
def extract_feature(img, color_space="RGB", spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                    cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than "RGB"
    if color_space != "RGB":
        if color_space == "HSV":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == "LUV":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == "HLS":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == "YUV":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == "YCrCb":
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == "ALL":
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(
                    get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False,
                                     feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def find_cars(img, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
              hist_bins):
    """
    Extract features using hog sub-sampling to make predictions
    :param img:
    :param y_start:
    :param y_stop:
    :param scale:
    :param svc:
    :param X_scaler:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param spatial_size:
    :param hist_bins:
    :return: Array of boxes and the corresponding heatmap
    """
    t = time.time()
    img_boxes = []
    count = 0
    # Make a heatmap of zeros
    heatmap = np.zeros_like(img[:, :, 0])

    img_tosearch = img[y_start:y_stop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv="RGB2YCrCb")
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    x_blocks = (ch1.shape[1] // pix_per_cell) - 1
    y_blocks = (ch1.shape[0] // pix_per_cell) - 1
    features_per_block = orient * cell_per_block ** 2
    window = 64
    blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    x_steps = (x_blocks - blocks_per_window) // cells_per_step
    y_steps = (y_blocks - blocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(x_steps):
        for yb in range(y_steps):
            count += 1
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[y_pos:y_pos + blocks_per_window, x_pos:x_pos + blocks_per_window].ravel()
            hog_feat2 = hog2[y_pos:y_pos + blocks_per_window, x_pos:x_pos + blocks_per_window].ravel()
            hog_feat3 = hog3[y_pos:y_pos + blocks_per_window, x_pos:x_pos + blocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract the image patch
            sub_img = cv2.resize(ctrans_tosearch[y_top:y_top + window, x_left:x_left + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(sub_img, size=spatial_size)
            hist_features = color_hist(sub_img, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(x_left * scale)
                ytop_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                bboxes = (xbox_left, (ytop_draw + y_start)), ((xbox_left + win_draw), (ytop_draw + win_draw + y_start))
                img_boxes.append(bboxes)
                # heatmap[ytop_draw + y_start:ytop_draw + win_draw + y_start, xbox_left: xbox_left + win_draw] += 1
    add_heat(heatmap, img_boxes)
    # print("Runtime={:.2f}s, total windows={}".format((time.time() - t), count))
    return img_boxes, heatmap


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes
