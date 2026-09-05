import cv2

FILE_NAME = "image5.jpg"
# Load the color image from the images folder
image = cv2.imread(f"images/{FILE_NAME}")

# Convert the image from BGR (OpenCV default) to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the converted grayscale image to disk
cv2.imwrite(f"modified/mod_{FILE_NAME}", gray)

# Display the original and grayscale images in separate windows
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
  
# Wait for a key press and then close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()