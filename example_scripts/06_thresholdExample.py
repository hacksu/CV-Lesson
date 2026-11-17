import cv2
import numpy as np

#Read the image
image = cv2.imread("notes.png")

#Convert the Image to a grayscale
#Use BGR to Gray to match imread
#This is needed as thresholding is based on individual pixel intensity, not one specific color channel
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Binary Thresh
#Values under the threshold are converted to black while above are turned white
ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("thresh.png", thresh)

#Apply Otsu Method Threshing, better for images with 2 pixel peaks, automatically tries to calculate the best threshold by minimizing intraclass differences
#0 gets ignored
optimal_thresh, otsu_thresh = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite("otsu_thresh.png", otsu_thresh)

#Apply Adaptive Threshold, segments image out into subgroups and calculates threshold for each subgroup separately
adap_thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 6)

cv2.imwrite("adap_thresh.png", adap_thresh)