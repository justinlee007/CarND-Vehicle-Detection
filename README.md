#Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project utilizes a software pipeline to classify vehicles and track them in a video.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window or sub-sampling technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The images for classification are in `vehicles` and `non_vehicles` symlink.  Because the symlinks point to a location outside the repository, the mechanism for training the classifier cannot be used unless the images are downloaded from the class project site.  The images in `test_images` are used in testing the pipeline on single frames.  Examples of the output from each stage of the pipeline are the `ouput_images` folder.  The video `vehicle_detection.mp4` is target video for the lane-finding pipeline.  Each rubric step will be documented with output images and usage.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/spatial_bin_cutout6.png
[image3]: ./output_images/color_histogram_cutout2.png
[image4]: ./output_images/hog_features_31.png
[image5]: ./output_images/heatmap_test6.png
[image6]: ./output_images/heatmap_label_test6.png
[image7]: ./output_images/heatmap_thresh_label_test6.png
[video1]: https://youtu.be/qrLMPmFXFF8

# [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)

##Feature Extraction
###1. Explain how (and identify where in your code) you extracted features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  This code is in the `trainer.load_features` method.  The file `car_notcar` is a utility that loads the images and displays a sample one:
```
usage: car_notcar.py [-h] [-show] [-save]

Utility for visualizing car/not-car pair

optional arguments:
  -h, --help  show this help message and exit
  -show       Show first car/not-car pair
  -save       Save car/not-car pair image to disk
```
![][image1]

The methods used for feature extraction are in `detection.py`.  The four steps I use for feature extractions are:

1. Color transform
2. Spatial binning of image channels 
3. Color histogram of image channels
4. Image [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)

#### Color Transform

The first step for feature extraction is color transform.  I played with all the color variants discussed in the lecture `(RGB, HSV, LUV, HLS, YUV and YCrCb)`.  The best color space in terms of higher test accuracy in my experience was `YUV` and `YCrCb`.  I think this is because the `Y` channel in both color spaces produces a grayscale-like image that makes it easier to pick up edges.  The difference between `YUV` and `YCrCb` is marginal - mostly related to exact formulas.
 
The color conversion mechanism when training the model is contained in `detection.extract_features`.  The method used to convert colors during video processing is `detection.convert_color`.

#### Spatial Binning

The `detection.bin_spatial` method creates a stack of each channel of the image as a vector.  The parameter for spatial binning is a resize option that will (most often) downsample the channel before raveling and adding to `hstack`.  I had the most success with spatial binning size of 40 or 32.  

The `spatial_bin` utility visualizes a sample image and it's corresponding spatially-binned feature plot:
```
usage: spatial_bin.py [-h] [-show] [-save]

Utility for visualizing spatial bin feature vector

optional arguments:
  -h, --help  show this help message and exit
  -show       Show car image with spatially binned plot
  -save       Save example spatial bin to disk
```
![][image2]
#### Color Histogram

The `detection.color_hist` method creates a stack of histograms of each image channel as a vector.  The parameters for color histogram include an image bin value for the number of equal-width bins in the given range. 
 
The `color_histogram` utility visualizes a sample image with it's corresponding historam plot per channel:

```
usage: color_histogram.py [-h] [-show] [-save]

Utility for visualizing color histogram feature extraction

optional arguments:
  -h, --help  show this help message and exit
  -show       Show each channel histogram
  -save       Save histogram visualization
```
![][image3]

#### HOG

The code for this step is contained in `detection.get_hog_features` method which calls `skimage.feature.hog` to extract features.  I explored different color spaces and different `skimage.hog` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

The `get_hog` utility visualizes a sample car image with it's corresponding HOG feature plot:
```
usage: get_hog.py [-h] [-show] [-save]

Utility for visualizing HOG feature extraction

optional arguments:
  -h, --help  show this help message and exit
  -show       Visualize HOG image
  -save       Save HOG image
```
![][image4]
#####This example HOG feature plot uses the `YCrCb` color space with parameters of `orient=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

###2. Explain how you settled on your final choice of HOG parameters.

The bin spatial parameters are as follows:

1. spatial size: a resize option that will (most often) downsample the channel before raveling

Earlier versions of bin spatial method included a color transform, but that was refactored out into `detection.extract_feature`.  For spatial size, I had the most success with spatial binning size of 40 or 32.  I stuck with 32 for my submission because 64 (the training image size) divides evenly into it.  

The color histogram parameters are as follows:

1. image bins: the number of equal-width bins in the given range

I had success classifying images with image bins of 72 or 32.  I stuck with 32 for my submission because 64 (the training image size) divides evenly into it.

The HOG parameters are as follows:

1. orient: the number of orientation bins
2. pixels per cell: the size (in pixels) of a cell as tuple (int, int) 
3. cell per block: the number of cells in each block as a tuple (int, int) 

I tried various combinations of parameters and had some success with parameters: 

`(orient = 9, pix_per_cell = (16, 16), cell_per_block = (2, 2))` 

However, I wanted to use the subsample feature as a sliding window mechanism later in the project so I decreased the pix_per_cell to account for this.  My project submission uses: 

`(orient = 9, pix_per_cell = (8, 8), cell_per_block = (2, 2))`

###3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Classifier training occurs in `trainer.train_classifier` in the following order:

1. Car and non-car images are loaded into memory
2. Features are extracted using the process above into vector (`vstack`)
3. A per-column scaler is fit to the vector for normalization
4. A label vector is created and initialized using the length of the car and non-car image sets (with 1s for cars and 0s for non-cars)
5. The data is split into training and test sets with 20% as the test size
6. A LinearSVC then trains the model on the training set
7. The model is tested on the test set and the score is displayed
8. If specified, the model is saved (along with all relevant training parameters) as a pickled file on the disk

The `trainer` file usage is as follows:
```
usage: trainer.py [-h] [-save]

Utility for training and storing SVC

optional arguments:
  -h, --help  show this help message and exit
  -save       Pickle SVC, Scalar and training parameters
```
Here is the console output when I trained my model:
```
len(cars)=8792, len(notcars)=8968
orient=9, pix_per_cell=8, cell_per_block=2 
Feature extraction:		117.85s
Feature vector length:	8460
SVC training time:		23.42s
Test accuracy:			0.9913
Saving to file:			svc_pickle.p

Process finished with exit code 0
```

##Sliding Window Search

###1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented both sliding multi-scale window search and a HOG sub-sampling window search.  My submission includes only the HOG sub-sampling search because it performed much faster and lowered training time drastically.

####Multi-window sliding search
For the window search, the following parameters are used:

1. size: a square search window size
2. y-start: starting pixel location on the y-axis (used to avoid searching the sky)
3. y-stop: stop pixel location on the y-axes (used to avoid searching the hood)
4. overlap: percentange of overlap between the search windows

In my testing for the multi-scale window search, I used the following scales and windowing scheme:

| size | y-start | y-stop | overlap |
| :---: | :---: | :---: | :---: |
| (64, 64) | 400 | y - 128 | 0.5 |
| (128, 128) | 400 | y - 64 | 0.5 |
| (256, 256) | 336 | y | 0.5 |

My tests for the multi-scale window search was satisfactory for the above parameters, but performed slowly.  

####HOG sub-sampling search
For HOG sub-sampling, the window size is fixed at (64, 64) and the sub-sampling is performed at scales (or multipliers) of the base window size.  The following parameters were used:

1. y-start: starting pixel location on the y-axis (used to avoid searching the sky)
2. y-stop: stop pixel location on the y-axes (used to avoid searching the hood)
3. scale: the multiplier for the base (64, 64) window size

I tested scales of 1, 1.5 and 2.  I also tested running the HOG sub-sampler multiple times at different scales.  I found that the increase in vehicle detection and tracking was negligible for the extra scaling runs so my submission only uses scale of 1.5, y-start of 400 and y-stop of (y - 64)


###2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

The `heatmap` utility visualizes the complete pipeline:
```
usage: heatmap.py [-h] [-show] [-save] [-thresh] [-label]

Utility for visualizing vehicle detection heatmap

optional arguments:
  -h, --help  show this help message and exit
  -show       Show vehicle detection boxes with corresponding heatmap
  -save       Save example image to disk
  -thresh     Apply threshold value of 2 for heatmap
  -label      Use heatmap-based labeled bounding boxes
```
![][image5]
#####HOG sub-sampling with heatmap using bounding boxes (no threshold, no label)
## Video Implementation

###1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Video processing functionality is in `pipeline`, where the test video can be specified as a command-line arg:

```
usage: pipeline.py [-h] [-test]

Main entry for vehicle detection project

optional arguments:
  -h, --help  show this help message and exit
  -test       Use test video
```


Here's a [link to my video result][video1]


###2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The `heatmap` utility from above allows for visualizing the difference between using just bounding boxes with usage of `scipy.ndimage.measurements.label` function and also thresholding:

![alt text][image6]
#####HOG sub-sampling with heatmap using `measurements.label` for bounding box

![alt text][image7]
#####HOG sub-sampling with heatmap using `measurements.label` for bounding box with thresholding

####Bounding box cache
Finally, I created `VehicleDetector` class to load the SVC from disk only once during video processing.  This class also used a deque data structure to save the 5 previous sets of image boxes.  For each video frame, while preparing to draw new boxes, the heatmap would be summed through the set of previous bounding boxes.  In this way, I was able to raise the threshold value to 4 and remove much more false positives.  

##Discussion

###1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

  

