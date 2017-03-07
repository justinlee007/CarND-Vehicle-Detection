import collections

import cv2
import numpy as np
from scipy.ndimage.measurements import label

from detection import add_heat
from detection import find_cars
from trainer import load_svc


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
        self.hist_queue = collections.deque([], maxlen=5)
        self.box_queue = collections.deque([], maxlen=5)

    def init_svc(self):
        self.svc, self.X_scaler, self.orient, self.pix_per_cell, \
        self.cell_per_block, self.spatial_size, self.hist_bins = load_svc()

    def find_cars(self, image, scale):
        if self.initialized is False:
            self.init_svc()
            self.initialized = True
        y = image.shape[0]
        y_start = 400
        y_stop = y - 64
        return find_cars(image, y_start, y_stop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell,
                         self.cell_per_block, self.spatial_size, self.hist_bins)

    # Define a function to draw bounding boxes
    def draw_boxes(self, img, bboxes, color=(0, 0, 1), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    def draw_labeled_bboxes(self, img, labels, color=(0, 0, 1), thick=6):
        # Iterate through all detected cars
        bboxes = []
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = (np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))
            bboxes.append(bbox)
            # Draw the boxes on the image
        return self.draw_boxes(img, bboxes, color, thick)

    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def process_image(self, image):
        scale = 1.5

        # Uncomment the following line if you extracted training data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        draw_image = image.astype(np.float32) / 255
        # print("min={}, max={}".format(np.min(draw_image), np.max(draw_image)))

        img_boxes, heatmap = self.find_cars(draw_image, scale)

        self.hist_queue.append(heatmap)
        self.box_queue.append(img_boxes)

        all_heat = np.zeros_like(image[:, :, 0])

        for boxes in self.box_queue:
            add_heat(all_heat, boxes)

        # heatmap = self.apply_threshold(heatmap, 2)
        all_heat = self.apply_threshold(all_heat, 4)

        # heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(all_heat)
        image = self.draw_labeled_bboxes(image, labels, color=(0, 0, 255))
        return image
