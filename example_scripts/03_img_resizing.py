import cv2
import numpy as np

#Read the image (Open CV loads images in B,G,R color format)
image = cv2.imread("cat.jpg")

#Shape of the original image (H, W, C)
#Notice image.shape shows height first because openCV prides themselves on being just a little infuriating
print(f"Original Image: {image.shape}")

#Calculates the aspect ratio of the image for resizing
aspect_ratio = min(256 / image.shape[1], 256 / image.shape[0])

new_size = int(image.shape[1] * aspect_ratio), int(image.shape[0] * aspect_ratio)
small_size = int(image.shape[1] * aspect_ratio * 0.25), int(image.shape[0] * aspect_ratio * 0.25)

#Notice how this takes Width x Height, thanks openCV
resized_img = cv2.resize(image, new_size)
super_small_img = cv2.resize(image, small_size)

#Print new images sizes
print(f"Medium Image: {resized_img.shape}")
print(f"Small Image: {super_small_img.shape}")

#Write images to disk
cv2.imwrite("256x256.png", resized_img)
cv2.imwrite("64x60.png", super_small_img)


#Types of Interpolation
#Warning, makes some kinda large images
y, x, size = 170, 390, 48
crop = image[y:y+size, x:x+size]
larger_size = int(image.shape[1] * 3), int(image.shape[0] * 3)

#Nearest Neighbor
#Predicts the pixel based on the nearest pixel
nearest_neighbor = cv2.resize(crop, larger_size, interpolation = cv2.INTER_NEAREST)

#BiLinear
#Predicts the pixel based on the average of the 4 nearest
bi_linear = cv2.resize(crop, larger_size, interpolation = cv2.INTER_LINEAR)

#BiCubic
#Predicts the pixel based on the average of the 16 nearest
bi_cubic = cv2.resize(crop, larger_size, interpolation = cv2.INTER_CUBIC)

cv2.imwrite("nearest_neighbor.png", nearest_neighbor)
cv2.imwrite("bi_linear.png", bi_linear)
cv2.imwrite("bi_cubic.png", bi_cubic)