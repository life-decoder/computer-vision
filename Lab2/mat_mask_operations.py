from __future__ import print_function
import sys
import time
import numpy as np
import cv2 as cv
def is_grayscale(my_image):
    # Grayscale images have shape (H, W), while color images have (H, W, C).
    return len(my_image.shape) < 3
def saturated(sum_value):
    # Clamp computed pixel values to valid 8-bit range.
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value
def sharpen(my_image):
    if is_grayscale(my_image):
        height, width = my_image.shape
    else:
        # Ensure color input is in 8-bit format before per-channel processing.
        my_image = cv.cvtColor(my_image, cv.CV_8U)
        height, width, n_channels = my_image.shape
    # Output image with same shape and dtype as input.
    result = np.zeros(my_image.shape, my_image.dtype)
    
    # Skip border pixels because the 3x3 neighborhood is not fully available there.
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if is_grayscale(my_image):
                # 3x3 sharpening mask: center*5 - top - bottom - left - right.
                sum_value = 5 * my_image[j, i] - my_image[j + 1, i] - my_image[j - 1, i] \
                            - my_image[j, i + 1] - my_image[j, i - 1]
                result[j, i] = saturated(sum_value)
            else:
                # Apply the same sharpening formula independently per color channel.
                for k in range(0, n_channels):
                    sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k]  \
                                - my_image[j - 1, i, k] - my_image[j, i + 1, k]\
                                - my_image[j, i - 1, k]
                    result[j, i, k] = saturated(sum_value)
    
    return result
def main(argv):
    filename = 'lena_gray.jpg' #input image data in unsigned 8 bit format
    img_codec = cv.IMREAD_COLOR
    if argv:
        filename = sys.argv[1]
        # Optional second argument "G" forces grayscale loading.
        if len(argv) >= 2 and sys.argv[2] == "G":
            img_codec = cv.IMREAD_GRAYSCALE
    src = cv.imread(cv.samples.findFile(filename), img_codec)
    if src is None:
        print("Can't open image [" + filename + "]")
        print("Usage:")
        print("mat_mask_operations.py [image_path -- default lena.jpg] [G -- grayscale]")
        return -1
    cv.namedWindow("Input", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Output", cv.WINDOW_AUTOSIZE)
    cv.imshow("Input", src)
    # Measure runtime of the manual implementation.
    t = round(time.time())
    dst0 = sharpen(src)
    t = (time.time() - t)
    print("Hand written function time passed in seconds: %s" % t)
    cv.imshow("Output", dst0)
    cv.waitKey()
    t = time.time()
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)  # kernel should be floating point type
    
    # Apply the equivalent built-in convolution for comparison.
    dst1 = cv.filter2D(src, -1, kernel)
    # ddepth = -1, means destination image has depth same as input image
    
    t = (time.time() - t)
    print("Built-in filter2D time passed in seconds:     %s" % t)
    cv.imshow("Output", dst1)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])