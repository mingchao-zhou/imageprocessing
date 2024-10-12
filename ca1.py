import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the RGB image
image = cv2.imread('C:/Users/mingz/Documents/vscode/ca01/test.jpg') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# (a) Display the RGB image and its R, G, B channels separately

r_channel = image_rgb[:, :, 0]
g_channel = image_rgb[:, :, 1]
b_channel = image_rgb[:, :, 2]

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('RGB Image')

plt.subplot(2, 2, 2)
plt.imshow(r_channel, cmap='gray')
plt.title('Red Channel')

plt.subplot(2, 2, 3)
plt.imshow(g_channel, cmap='gray')
plt.title('Green Channel')

plt.subplot(2, 2, 4)
plt.imshow(b_channel, cmap='gray')
plt.title('Blue Channel')

plt.tight_layout()
plt.savefig('outcome_RGB_channel.png') 
plt.show()

# (b) Convert the RGB image to the HSV colorspace and display H, S, V channels separately

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h_channel = image_hsv[:, :, 0]
s_channel = image_hsv[:, :, 1]
v_channel = image_hsv[:, :, 2]

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original RGB Image')

plt.subplot(2, 2, 2)
plt.imshow(h_channel, cmap='gray')
plt.title('Hue Channel')

plt.subplot(2, 2, 3)
plt.imshow(s_channel, cmap='gray')
plt.title('Saturation Channel')

plt.subplot(2, 2, 4)
plt.imshow(v_channel, cmap='gray')
plt.title('Value Channel')

plt.tight_layout()
plt.savefig('outcome_HSV_channel.png')
plt.show()


# (c) Detect blue pixels with hue in range [110, 130]

lo_blue = np.array([110, 50, 50])
up_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(image_hsv, lo_blue, up_blue)
blue_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=blue_mask)

# Display the blue mask and blue pixels
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(blue_mask, cmap='gray')
plt.title('Blue Pixel Mask')

plt.subplot(1, 2, 2)
plt.imshow(blue_pixels)
plt.title('Detected Blue Pixels')

plt.tight_layout()
plt.savefig('outcome_blue_pixels.png')
plt.show()

# Load the low-contrast grayscale image
gray_image = cv2.imread('C:/Users/mingz/Documents/vscode/ca01/low_contrast_image.png', cv2.IMREAD_GRAYSCALE)

# (a) Display the low-contrast grayscale image

plt.imshow(gray_image, cmap='gray')
plt.title('Low-Contrast Grayscale Image')
plt.savefig('outcome_gray.png')
plt.show()

# (b) Calculate and plot the histogram

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title('Histogram of Low-Contrast Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('outcome_histogram.png')
plt.show()

# (c) Compute and plot the CDF of the histogram

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(gray_image.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.title('CDF and Histogram of the Image')
plt.savefig('outcome_cdf.png')
plt.show()

# (d) Apply histogram equalization using the CDF

equalized_image = cv2.equalizeHist(gray_image)

# Display the equalized image
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.savefig('outcome_equalized.png')
plt.show()

# Display the histogram of the equalized image
hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
plt.plot(hist_eq)
plt.title('Histogram of Equalized Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('outcome_histogram_equalized.png')
plt.show()

# show all
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(hist)
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
plt.plot(hist_eq)
plt.title('Equalized Histogram')

plt.tight_layout()
plt.savefig('outcome_histogram_all.png')
plt.show()
