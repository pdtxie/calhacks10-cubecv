import cv2


def find_points(lines, img):  # top 3 lines
    lines = list(map(lambda x: x[0], lines))

    rightmost = max(lines, key=lambda x: max(x[0], x[2]))
    bottommost = max(lines, key=lambda x: max(x[1], x[3]))

    c = tuple(map(int, (rightmost[2], rightmost[3])))
    d = tuple(map(int, (rightmost[0], rightmost[1])))

    a = tuple(map(int, (bottommost[2], bottommost[3]))) 
    b = tuple(map(int, (bottommost[0], bottommost[1])))

    # magenta
    cv2.imshow("1", cv2.circle(img, a, 1, (255, 0, 255), 2))
    cv2.imshow("1", cv2.circle(img, b, 1, (255, 0, 255), 2))
    # cyan
    cv2.imshow("2", cv2.circle(img, c, 1, (0, 255, 255), 2))
    cv2.imshow("2", cv2.circle(img, d, 1, (0, 255, 255), 2))

