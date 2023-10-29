import cv2 
import numpy as np 

green = [[43, 153, 0], [132, 255, 255], 'green']
blue = [[100, 143, 145], [118, 255, 255], 'blue']

orange = [[0, 172, 83], [20, 255, 255], 'orange']
red = [[170, 145, 80], [179, 255, 255], 'red']

white = [[60, 60, 215], [120, 115, 255], 'white']
yellow = [[28, 145, 0], [40, 255, 255], 'yellow']

ranges = [green, blue, orange, red, white, yellow]


USE_CAM = True

if USE_CAM:
    cap = cv2.VideoCapture(1)
    waitTime = 330
else:
    img = cv2.imread('media/test_1.jpg')


while (True):
    if USE_CAM:
        ret, img = cap.read()

    scale = 4
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
        # cv2.imshow(f'color {color[2]}', detected)

    cv2.imshow('final image before', final_image)

    cv2.imshow("blur", cv2.GaussianBlur(final_image, (5, 5), 0))

    final_image = cv2.Canny(image=cv2.GaussianBlur(final_image, (5, 5), 0), threshold1=100, threshold2=200)

    cv2.imshow('final image', final_image)

# im2, contours, hierarchy = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# convex_hull = cv2.convexHull(final_image)
    if USE_CAM and cv2.waitKey(waitTime) & 0xFF == ord('q'):
        break

# Remove noise
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and find total area
# cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# cv2.drawContours(original, cnts, -1, (0,0,0), 2)

# cv2.imshow('mask', mask)
# cv2.imshow('opening', opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
