# Simple script to load and display an image using OpenCV
import argparse
import cv2

# Parse command-line arguments
ap = argparse.ArgumentParser()
# Expect an input image path via -i or --image
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())  # Convert Namespace to dict for convenient access

# Read the image from disk (returns None if file not found)
image = cv2.imread(args["image"])

# Display the image in a window titled "image"
cv2.imshow("image", image)
# Wait indefinitely for a key press, then close the window
cv2.waitKey(0)