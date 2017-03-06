import matplotlib

matplotlib.use('TkAgg')
from trainer import *

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
