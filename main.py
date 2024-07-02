from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
 
rng.seed(12345)
 
 
parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance Transform and Watershed Algorithm.\
 Sample code showing how to segment overlapping objects using Laplacian filtering, \
 in addition to Watershed and Distance Transformation')
parser.add_argument('--input', help='Path to input image.', default='cards.png')
args = parser.parse_args()
 
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
 print('Could not open or find the image:', args.input)
 exit(0)
 
# Show source image
cv.imshow('Source Image', src)
 
 
 
src[np.all(src == 255, axis=2)] = 0
 
# Show output image
cv.imshow('Black Background Image', src)
 
 
 
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
 
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian
 
# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)
 
#cv.imshow('Laplace Filtered Image', imgLaplacian)
cv.imshow('New Sharped Image', imgResult)
 
 
 
bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
cv.imshow('Binary Image', bw)
 
 
 
dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
 
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', dist)
 
 
 
_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
 
# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv.dilate(dist, kernel1)
cv.imshow('Peaks', dist)
 
 
 
dist_8u = dist.astype('uint8')
 
# Find total markers
contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
 
# Create the marker image for the watershed algorithm
markers = np.zeros(dist.shape, dtype=np.int32)
 
# Draw the foreground markers
for i in range(len(contours)):
 cv.drawContours(markers, contours, i, (i+1), -1)
 
# Draw the background marker
cv.circle(markers, (5,5), 3, (255,255,255), -1)
markers_8u = (markers * 10).astype('uint8')
cv.imshow('Markers', markers_8u)
 
 
 
cv.watershed(imgResult, markers)
 
#mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype('uint8')
mark = cv.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
#cv.imshow('Markers_v2', mark)
 
# Generate random colors
colors = []
for contour in contours:
 colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
 
# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
 
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
 for j in range(markers.shape[1]):
 index = markers[i,j]
 if index > 0 and index <= len(contours):
 dst[i,j,:] = colors[index-1]
 
# Visualize the final image
cv.imshow('Final Result', dst)
 
 
cv.waitKey()


# Explanation for this code 
# Imports necessary libraries: cv2 for image processing, numpy for numerical operations, argparse for handling command-line arguments, and random for generating randome colors. 
# Sets a seed for the random number generator to ensure consistent results across runs.
# Parses command-line arguments using argparse. The script expects an --input argument specifying the path of the image file.


# Loading and Preprocessing.
# Reads the image using cv2.imread
# Checks if the image is loaded successfully
# Shows the source image using cv.imshow
# Sets all pixels with all channels (RGB) equal to 255 (white) to black (0,0,0) to remove potential background noise.
# Shows the image with a black background


# Sharpening with Laplacian filter:
# Defines a Laplacian filter kernel for sharpening the image.
# Applies the Laplacian filter using cv2.filter2D
# Converts the source image to float32 for calculations with negative values.
# Subtracts the Laplacian image from the source image to enhance edges
# Clips the resulting image values between 0 and 255 to fit the uint8 format
# Converts the images back to uint8 for display
# Shows the sharpened image (commented out)


# Binarization:
# Converts the sharpened image to grayscale using cv2.cvtColor
# Applies Otsu's thresholding to convert the grayscale image to binary (black and white) using cv2.threshold
#Shows the binary image


# Distance Transform and Thresholding
# Calculates the distance transform of the binary image using cv2.distanceTransform. This creates an image where each pixel value represents the distance to the nearest background pixel.
# Normalizes the distance image to a range of 0.0 to 1.0 for visualization and thresholding using cv2.normalize
# Shows the distance transform image.
# Applies thresholding on the distance image to identify foreground object peaks using cv2.threshold
# Dilates the thresholded distance image using a 3x3 kernel to slightly enlarge the foreground object markers using cv2.dilate
# Shows the image with identified peaks (foreground markers)


# Marker Preparation:
# Converts the dilated distance image to uint8 format using astype
# Finds contours in the distance image using cv2.findContours. Contours represent the boundaries of foreground objects.
# Creates a marker image with the same size and data type as the distance image using np.zeros. This image will be used to guide the watershed algorithm.
# Iterates through the contours and draws them on the marker image using cv2.drawContours. Each contour is assigned a unique marker value (i+1)
# Draws a background marker (value 255) at a specific location (5,5) using cv2.circle. This marker indicates the bakcground region.
# Converts the marker image to uint8 format for compatibility with the watershed algorithm.
# Shows the marker image with different colors for each foreground object.


# Watershed Segmentation
# Applies the watershed algorithm using cv2.watershed with the image and the marker image as input. The watershed algorithm segments the image based on the markers and the distance transform information. 
# Creates a new image to store the segmentation results.
# Inverts the marker image using bitwise NOT operation (cv2.bitwise_not). This step is likly commented out as it doesn't affect the final result significantly. (commented out section)


# Coloring and Visualization:
# Generates a list of random colors for each foreground object using a loop
# Fills the segmented regions in the result image with the corresponding random colors based on the marker values.
# Shows the final segmented image with each object displayed in a different color


# Waiting for Key Press:
# Press for a key press using cv2.waitKey to keep the program running until the user closes the window. 

