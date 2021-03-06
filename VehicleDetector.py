import collections

import numpy as np
import scipy.ndimage.measurements as measurements

import detection
import trainer


class VehicleDetector:
    def __init__(self):
        self.initialized = False
        self.svc = None
        self.X_scaler = None
        self.orient = None
        self.pix_per_cell = None
        self.cell_per_block = None
        self.spatial_size = None
        self.hist_bins = None
        self.box_deques = [collections.deque([], maxlen=5), collections.deque([], maxlen=5)]
        self.scales = [1.5, 2]

    def init_svc(self):
        self.svc, self.X_scaler, self.orient, self.pix_per_cell, \
        self.cell_per_block, self.spatial_size, self.hist_bins = trainer.load_svc()

    def find_cars(self, image, scale):
        if self.initialized is False:
            self.init_svc()
            self.initialized = True
        y = image.shape[0]
        y_start = 400
        y_stop = y - 64
        return detection.find_cars(image, y_start, y_stop, scale, self.svc, self.X_scaler, self.orient,
                                   self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)

    def process_image(self, image):
        # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        draw_image = image.astype(np.float32) / 255
        # print("min={}, max={}".format(np.min(draw_image), np.max(draw_image)))

        combined_heatmap = np.zeros_like(image[:, :, 0])
        for index, scale in enumerate(self.scales):
            box_deque = self.box_deques[index]

            img_boxes, heatmap = self.find_cars(draw_image, scale)
            box_deque.append(img_boxes)

            for boxes in box_deque:
                detection.add_heat(combined_heatmap, boxes)

        combined_heatmap = detection.apply_threshold(combined_heatmap, 4)

        # Find final boxes from heatmap using label function
        labels = measurements.label(combined_heatmap)
        image = detection.draw_labeled_bboxes(image, labels, color=(0, 0, 255))
        return image
