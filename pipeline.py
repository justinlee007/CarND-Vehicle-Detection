from detection_functions import *
from hog_subsample import *

def process_image(img):
    out_img, heat_map = find_cars()