import cv2

#Read the image (Open CV loads images in B,G,R color format)
image_bgr = cv2.imread("cat.jpg")

print(image_bgr.shape)
print(image_bgr.dtype)
