#Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project utilizes a software pipeline to classify vehicles and track them in a video.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The images for classification are in `vehicles` and `non_vehicles` symlink.  The images in `test_images` are used in testing the pipeline on single frames.  Examples of the output from each stage of the pipeline are the `ouput_images` folder.  The video `vehicle_detection.mp4` is target video for the lane-finding pipeline.  Each rubric step will be documented with output images and usage.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/spatial_bin_cutout6.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: https://youtu.be/qrLMPmFXFF8

# [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

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
![alt text][image1]

The methods used for feature extraction are in `detection.py`.  The four steps I use for feature extractions are:
1) Color transform
2) Spatial binning of image channels 
3) Color histogram of image channels
4) Image [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)

#### Color Transform

The first step for feature extraction is color transform.  I played with all the color variants discussed in the lecture `(RGB, HSV, LUV, HLS, YUV and YCrCb)`.  The best color space in terms of higher test accuracy in my experience was `YUV` and `YCrCb`.  I think this is because the `Y` channel in both color spaces produces a grayscale-like image that makes it easier to pick up edges.  The difference between `YUV` and `YCrCb` is marginal - mostly related to exact formulas. 

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
![alt text][image2]
#### Color Histogram

The `detection.color_hist` method creates a stack of histograms of each image channel as a vector.  The parameters for color histogram are 
#### HOG
Next, I use a color histogram feature vector is extracted

The code for this step is contained in `detection.py` where the method `get_hog_features` will call `skimage.feature.hog` to extract features.  



I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

###2. Explain how you settled on your final choice of HOG parameters.

The bin spatial parameters are as follows:
1) color space:
2) spatial size: 

The color histogram parameters are as follows:
2) image bins: the number of equal-width bins in the given range

The HOG parameters are as follows:
1) orient
2) pixels_per_cell
3) cell_per_block

spatial = 40
histbin = 72

I tried various combinations of parameters and...

###3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

```
usage: trainer.py [-h] [-save]

Utility for training and storing SVC

optional arguments:
  -h, --help  show this help message and exit
  -save       Pickle SVC, Scalar and training parameters
```
I trained a linear SVM using...
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

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

###2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

## Video Implementation

###1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


###2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


![alt text][image5]
##### Here are six frames and their corresponding heatmaps:

![alt text][image6]
##### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![alt text][image7]
##### Here the resulting bounding boxes are drawn onto the last frame in the series:

##Discussion

###1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

