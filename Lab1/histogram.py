import cv2
import numpy as np
from matplotlib import pyplot as plt

# An image histogram shows the distribution of pixel intensities.
# It helps reveal brightness, contrast, and color balance in an image.

# Read the image from disk
img = cv2.imread('images/image1.jpg')

# Plot color histograms for B, G and R channels
color = ('b','g','r')
for i,col in enumerate(color):
   # Calculate histogram for channel i
   histr = cv2.calcHist([img],[i],None,[256],[0,256])
   # Plot the histogram using matplotlib
   plt.plot(histr,color = col)
   plt.xlim([0,256])

plt.show()