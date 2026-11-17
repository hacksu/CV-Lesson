import cv2
import numpy as np

#Read the image (Open CV loads images in B,G,R color format)
image_bgr = cv2.imread("cat.jpg")

#Split seperates these color channels out into 3 different channels
blue_chan, green_chan, red_chan = cv2.split(image_bgr)

#Create a blank (black) channel of the exact same height and width as the other channels
zeros = np.zeros_like(blue_chan)

#Merge each channel back with zeros in the other two slots (BGR order), isolating hues
blue_tinted = cv2.merge([blue_chan, zeros, zeros])
green_tinted = cv2.merge([zeros, green_chan, zeros])
red_tinted = cv2.merge([zeros, zeros, red_chan])

#Print a pixel of the image and the same pixel on each mask
#Notice how the 3 masks added together are the same as the pixel
print(f"Original Image Pixel: {image_bgr[0][0]}")
print(f"----------")
print(f"Blue Mask Pixel: {blue_tinted[0][0]}")
print(f"Green Mask Pixel: {green_tinted[0][0]}")
print(f"Red Mask Pixel: {red_tinted[0][0]}")

#Exagurates color masks, showing the colors off better
#Excess_red = (R-G) + (R-B), exagurates pixels where they are mostly green
excess_red = 2 * red_chan.astype(np.float32) - green_chan.astype(np.float32) - blue_chan.astype(np.float32)
excess_red = np.clip(excess_red, 0, 255).astype(np.uint8) #Clip binds our values back to 0-255

#Other Colors
excess_green = 2 * green_chan.astype(np.float32) - red_chan.astype(np.float32) - blue_chan.astype(np.float32)
excess_green = np.clip(excess_green, 0, 255).astype(np.uint8) #Clip binds our values back to 0-255

excess_blue = 2 * blue_chan.astype(np.float32) - red_chan.astype(np.float32) - green_chan.astype(np.float32)
excess_blue = np.clip(excess_blue, 0, 255).astype(np.uint8) #Clip binds our values back to 0-255




# cv2.imshow("Blue Channel", blue_chan)
# cv2.imshow("Green Channel", green_chan)
# cv2.imshow("Red Channel", red_chan)

# cv2.imshow("Blue Channel", blue_tinted)
# cv2.imshow("Green Channel", green_tinted)
# cv2.imshow("Red Channel", red_tinted)

#Write Images
cv2.imwrite("blue_channel.png", blue_chan)
cv2.imwrite("green_channel.png", green_chan)
cv2.imwrite("red_channel.png", red_chan)

cv2.imwrite("blue_tinted.png", blue_tinted)
cv2.imwrite("green_tinted.png", green_tinted)
cv2.imwrite("red_tinted.png", red_tinted)

cv2.imwrite("excess_red.png", excess_red)
cv2.imwrite("excess_green.png", excess_green)
cv2.imwrite("excess_blue.png", excess_blue)


