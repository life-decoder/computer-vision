import cv2
import matplotlib.pyplot as plt

# 1. Load a real-world noisy photograph in grayscale
image = cv2.imread("astronaut.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image")
    exit(-1)

# 2. Apply Gaussian smoothing (Task 6 parameter: kernel size 5x5, sigmaX = 2)
gaussian_blur = cv2.GaussianBlur(image, (5, 5), sigmaX=60)

# 3. Apply Mean filter on the same image (kernel size 5x5)
mean_blur = cv2.blur(image, (20,20))

# 4. Display original, Gaussian, and Mean results together in 1 row x 3 columns
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title('Gaussian Smoothed')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mean_blur, cmap='gray')
plt.title('Mean Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()