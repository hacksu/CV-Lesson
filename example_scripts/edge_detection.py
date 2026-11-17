#Example of Sobel Edge Detection using Kernels
#Sobel uses 2 Kernels, one that detects horiziontal edges and one that detects vertical edges
#Then averages those together to do edge detection
import cv2

#Read Image
image = cv2.imread("cat.jpg")

#Convert image to grayscale for edge detection
#Grays work far better as edges are based on lighting differences.
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply the Sobel Kernels to the image
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute gradient magnitude
#Calculated by sqrt(x^2 + y^2)
gradient_magnitude = cv2.magnitude(sobelx, sobely)
 
# Convert to uint8
#Needed for display
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
 
# Display result
cv2.imwrite("sobelDetection.png", gradient_magnitude)


#Also test against a blurred image that reduces noise
#Box blur with 5x5 kernel
blurred_image = cv2.blur(image, (5,5))

sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Compute gradient magnitude
#Calculated by sqrt(x^2 + y^2)
blurred_gradient_magnitude = cv2.magnitude(sobelx, sobely)
 
# Convert to uint8
#Needed for display
blurred_gradient_magnitude = cv2.convertScaleAbs(blurred_gradient_magnitude)

cv2.imwrite("blurredSobelDetection.png", blurred_gradient_magnitude)


#Now use Canny algorithm
#To complex to explain here, but Canny is a more robust algorithm with a dedicated Gaussian blur

#Load image in grayscale
image = cv2.imread("cat.jpg", cv2.IMREAD_GRAYSCALE)

#Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(image, (5, 5), 1.4)

#Apply Canny Edge Detector
edges = cv2.Canny(blur, threshold1=50, threshold2=125)

#Save result
cv2.imwrite("canny.png", edges)