import cv2 
import numpy as np 

green = [[40, 55, 55], [70, 200, 200], 'green']
yellow = [[28, 145, 0], [40, 255, 255], 'yellow']
orange = [[0, 42, 200], [20, 255, 255], 'orange']
blue = [[100, 143, 145], [118, 255, 255], 'blue']
white = [[60, 60, 215], [120, 115, 255], 'white']
red = [[170, 145, 80], [179, 255, 255], 'red']
ranges = [green, yellow, orange, blue, white, red]

# Let's load a simple image with 3 black squares 
img = cv2.imread('test_4.jpg') 
scale = 10
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
cv2.imshow('original', img)

final_image = np.zeros(img.shape, dtype=np.uint8)

for color in ranges:
    image = img.copy()
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(color[0], dtype="uint8")
    upper = np.array(color[1], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    detected = cv2.bitwise_and(original, original, mask=mask)
    final_image = cv2.bitwise_or(final_image, detected)
    cv2.imshow(f'color {color[2]}', detected)

# Remove noise
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and find total area
# cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# cv2.drawContours(original, cnts, -1, (0,0,0), 2)

# cv2.imshow('mask', mask)
# cv2.imshow('opening', opening)
cv2.imshow(f'final image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
# Find Canny edges 
edged = cv2.Canny(image, 30, 200) 

# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged, 
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
cv2.imshow('Canny Edges After Contouring', edged) 
cv2.waitKey(0) 

print("Number of Contours found = " + str(len(contours))) 

# Draw all contours 
# -1 signifies drawing all contours 
for i in range(len(contours)):
    im2 = cv2.drawContours(image.copy(), contours, i, (255, 0, 0), 3) 
    cv2.imshow('Contours', im2)
    cv2.waitKey(0) 

cv2.destroyAllWindows() 
'''