import cv2
import numpy as np

#Read the image (Open CV loads images in B,G,R color format)
image = cv2.imread("cat.jpg")

#Convert the Image to a grayscale
#Use BGR to Gray to match imread
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_cat.png", gray_image)