import cv2 
import numpy as np 


USE_CAM = False
cap = None

GREEN = [[43, 153, 0], [132, 255, 255]]
BLUE = [[100, 143, 145], [118, 255, 255]]

ORANGE = [[0, 172, 83], [20, 255, 255]]
RED = [[170, 145, 80], [179, 255, 255]]

YELLOW = [[28, 145, 0], [40, 255, 255]]

RANGES = [GREEN, BLUE, ORANGE, RED, YELLOW]


if USE_CAM:
    cap = cv2.VideoCapture(1)
    waitTime = 330


def get_image():
    if USE_CAM:
        assert cap is not None, "cap not set"
        _, img = cap.read()
    else:
        img = cv2.imread('media/test_3.jpg')

    return img


def scale_image(img, scale=4):
    img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))

    return img


def mask_image(img):
    final_image = np.zeros(img.shape, dtype=np.uint8)

    for color in RANGES:
        image = img.copy()
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(color[0], dtype="uint8")
        upper = np.array(color[1], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        detected = cv2.bitwise_and(original, original, mask=mask)
        final_image = cv2.bitwise_or(final_image, detected)

    return final_image


def produce_contours(img):
    cannied = cv2.Canny(img, threshold1=200, threshold2=600)
    contours0, _ = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

    vis = np.zeros(img.shape, np.uint8)

    return cv2.drawContours(vis, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)


def fill_image(img):
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 240, cv2.THRESH_BINARY)[1]


def produce_image():
    img = get_image()
    img = scale_image(img)
    cv2.imshow("original", img)

    img = mask_image(img)
    cv2.imshow("masked", img)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    filled = fill_image(blurred)
    cv2.imshow("filled", filled)

    contours = produce_contours(img)
    cv2.imshow("contours", contours)


if USE_CAM:
    while True:
        produce_image()

        if USE_CAM and cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
else:
    produce_image()

cv2.waitKey(0)
cv2.destroyAllWindows()
