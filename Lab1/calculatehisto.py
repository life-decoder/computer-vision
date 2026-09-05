from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Code for Histogram Calculation tutorial.')
# Input image path (uses OpenCV sample finder by default)
parser.add_argument('--input', help='Path to input image.', default='images/image1.jpg')
args = parser.parse_args()

# Read source image
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Split image into B, G, R planes
bgr_planes = cv.split(src)

# Histogram parameters
histSize = 256
histRange = (0, 256) # the upper boundary is exclusive
# Whether to accumulate histogram across multiple calls. If True, new
# histogram values are added to the existing histogram; False computes
# a fresh histogram for this image (recommended here).
accumulate = False

# Calculate histograms for each channel
b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

# Prepare an image to draw the histograms
hist_w = 512
hist_h = 400
bin_w = int(round( hist_w/histSize )) #pixel width allocated to each histogram bin
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

# Normalize histogram heights to fit the display image
cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

# Draw lines for each bin for B, G and R channels
for i in range(1, histSize):
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(round(b_hist[i-1])) ),
            ( bin_w*(i), hist_h - int(round(b_hist[i])) ),
            ( 255, 0, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(round(g_hist[i-1])) ),
            ( bin_w*(i), hist_h - int(round(g_hist[i])) ),
            ( 0, 255, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(round(r_hist[i-1])) ),
            ( bin_w*(i), hist_h - int(round(r_hist[i])) ),
            ( 0, 0, 255), thickness=2)

# Show original and histogram image windows
cv.imshow('Source image', src)
cv.imshow('calcHist Demo', histImage)
cv.waitKey()
