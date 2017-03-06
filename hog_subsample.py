import pickle

import matplotlib

matplotlib.use('TkAgg')
import time
from detection_functions import *
from heatmap import *

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
                heatmap[ytop_draw + y_start:ytop_draw + win_draw + y_start, xbox_left: xbox_left + win_draw] += 1
    add_heat(heatmap, img_boxes)
    print("Runtime={:.2f}s, total windows={}".format((time.time() - t), count))
    return img_boxes, heatmap


def load_svc(save_file="svc_pickle.p"):
    print("Loading svc from {}".format(save_file))
    dist_pickle = pickle.load(open(save_file, "rb"))
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    print("orient={}, pix_per_cell={}, cell_per_block={}, spatial_size={}, hist_bins={}".format(orient, pix_per_cell,
                                                                                                cell_per_block,
                                                                                                spatial_size,
                                                                                                hist_bins))
    return svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins


if __name__ == '__main__':
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = load_svc()
    t = time.time()
    images = glob.glob("test_images/test*.jpg")
    images.append("test_images/bbox-example-image.jpg")
    for image_file in images:
        image = mpimg.imread(image_file)
        y = image.shape[0]

        # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        image = image.astype(np.float32) / 255
        print("min={}, max={}".format(np.min(image), np.max(image)))
        y_start = 400

        y_stop = y - 64

        scale = 2

        img_boxes, heatmap = find_cars(image, y_start, y_stop, scale, svc, X_scaler, orient, pix_per_cell,
                                       cell_per_block, spatial_size, hist_bins)
        if len(img_boxes) > 0:
            image = draw_boxes(image, img_boxes, color=(0, 0, 1), thick=6)

        plt.imshow(image)
        plt.show()

        plt.imshow(heatmap)
        plt.show()
