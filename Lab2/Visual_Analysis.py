import cv2
import matplotlib.pyplot as plt

# 1. Load image in grayscale
img = cv2.imread("ImageQ.jpg", cv2.IMREAD_GRAYSCALE)

# 2. Apply Gaussian Filters with varying sigma
# Gaussian Filter
# Assigns higher weights to pixels closer to the center kernel pixel. It provides smoother noise reduction while preserving edges better than mean filtering.
gaussian1 = cv2.GaussianBlur(img, (5, 5), 1)
gaussian3 = cv2.GaussianBlur(img, (5, 5), 3)
gaussian5 = cv2.GaussianBlur(img, (5, 5), 5)

# 3. Apply Mean Filters with corresponding kernel sizes
# Mean Filter
# Weighs all pixels within the kernel equally. This causes sharp uniform blurring and tends to blur edges and fine facial features more aggressively.
mean3 = cv2.blur(img, (3, 3))
mean5 = cv2.blur(img, (5, 5))
mean9 = cv2.blur(img, (9, 9))

# 4. Create figure layout: 2 Rows x 4 Columns
plt.figure(figsize=(12, 6))

# --- ROW 1: Gaussian Filters ---
plt.subplot(2, 4, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2, 4, 2), plt.imshow(gaussian1, cmap='gray'), plt.title('Gaussian σ=1')
plt.subplot(2, 4, 3), plt.imshow(gaussian3, cmap='gray'), plt.title('Gaussian σ=3')
plt.subplot(2, 4, 4), plt.imshow(gaussian5, cmap='gray'), plt.title('Gaussian σ=5')

# --- ROW 2: Mean Filters ---
plt.subplot(2, 4, 5), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2, 4, 6), plt.imshow(mean3, cmap='gray'), plt.title('Mean 3x3')
plt.subplot(2, 4, 7), plt.imshow(mean5, cmap='gray'), plt.title('Mean 5x5')
plt.subplot(2, 4, 8), plt.imshow(mean9, cmap='gray'), plt.title('Mean 9x9')

plt.tight_layout()  # Adjusts spacing between subplots to prevent overlapping
plt.show()