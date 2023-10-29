import cv2
from math import sqrt
import numpy as np
import sys
from algs import alg


USE_CAM = False
cap = None

GREEN = [[43, 153, 0], [132, 255, 255]]
BLUE = [[100, 143, 145], [118, 255, 255]]

ORANGE = [[0, 172, 83], [20, 255, 255]]
RED = [[170, 145, 80], [179, 255, 255]]

YELLOW = [[28, 145, 0], [40, 255, 255]]

RANGES = [GREEN, BLUE, ORANGE, RED, YELLOW]


if USE_CAM:
    cap = cv2.VideoCapture(0)
    waitTime = 330


def get_image():
    if USE_CAM:
        assert cap is not None, "cap not set"
        _, img = cap.read()
    else:
        img = cv2.imread(f'media/test_{sys.argv[1]}.jpg')

    return img


def scale_image(img, scale=4):
    img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))

    return img


def mask_image(img):
    final_image = np.zeros(img.shape, dtype=np.uint8)

    for colour in RANGES:
        image = img.copy()
        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(colour[0], dtype="uint8")
        upper = np.array(colour[1], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)
        detected = cv2.bitwise_and(original, original, mask=mask)
        final_image = cv2.bitwise_or(final_image, detected)

    return final_image


def setup_contours(img, epsilon):
    cannied = cv2.Canny(img, threshold1=200, threshold2=600)
    contours0, _ = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours0]

    return contours


def produce_contours(img, epsilon=10):
    contours = setup_contours(img, epsilon)

    vis = np.zeros(img.shape, np.uint8)

    return cv2.drawContours(vis, contours, -1, (255, 255, 255), 3, cv2.LINE_AA)


def produce_individual_contours(img) -> None:
    contours = setup_contours(img)

    vis = np.zeros(img.shape, np.uint8)

    for i, c in enumerate(sorted(contours, key=lambda x: cv2.contourArea(x))[-3:]):
        cv2.imshow(f"contour{i}", cv2.drawContours(vis.copy(), [c], 0, (255, 255, 255), 3, cv2.LINE_4))


def fill_image(img):
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 240, cv2.THRESH_BINARY)[1]


def connect(img):
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)

    for i in range(nlabels - 1):
        if areas[i] >= 5_000:
            result[labels == i + 1] = 255

    return result


def distance(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def detect_lines(img):
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]

    sorted_lines = sorted(lines, key=lambda x: distance(x[0][0], x[0][1], x[0][2], x[0][3]), reverse=True)
    best_lines = np.array(sorted_lines[:5])

    return best_lines, lsd.drawSegments(img, best_lines)


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

    # produce_individual_contours(img)

    connected_comps = connect(filled)
    cv2.imshow("connected", connected_comps)

    again = produce_contours(cv2.GaussianBlur(connected_comps, (3, 3), 0), epsilon=20)
    cv2.imshow("contours second", again)

    combined = cv2.bitwise_or(again, connected_comps)

    best_lines, with_lines = detect_lines(cv2.GaussianBlur(combined, (77, 77), 0))
    # best_lines, with_lines = detect_lines(combined)
    # best_lines, with_lines = detect_lines(again)
    cv2.imshow("with lines", with_lines)

    print(best_lines)
    alg.find_points(best_lines, with_lines)



if USE_CAM:
    while True:
        produce_image()

        if USE_CAM and cv2.waitKey(waitTime) & 0xFF == ord('q'):
            break
else:
    produce_image()

cv2.waitKey(0)
cv2.destroyAllWindows()
