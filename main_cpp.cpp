#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

bool USE_CAM = false;
VideoCapture cap;

vector<vector<int>> GREEN = {{43, 153, 0}, {132, 255, 255}};
vector<vector<int>> BLUE = {{100, 143, 145}, {118, 255, 255}};
vector<vector<int>> ORANGE = {{0, 172, 83}, {20, 255, 255}};
vector<vector<int>> RED = {{170, 145, 80}, {179, 255, 255}};
vector<vector<int>> YELLOW = {{28, 145, 0}, {40, 255, 255}};

vector<vector<vector<int>> > RANGES = {GREEN, BLUE, ORANGE, RED, YELLOW};

Mat get_image() {
    Mat img;
    if (USE_CAM) {
        if (!cap.isOpened()) {
            cerr << "Camera not initialized." << endl;
            return img;
        }
        cap >> img;
    } else {
        img = imread("media/test_4.jpg");
    }
    return img;
}

Mat scale_image(Mat img, int scale = 4) {
    Mat scaledImg;
    resize(img, scaledImg, Size(img.cols / scale, img.rows / scale));
    return scaledImg;
}

Mat mask_image(Mat img) {
    Mat final_image = Mat::zeros(img.size(), CV_8UC3);

    for (auto& colour : RANGES) {
        Mat image = img.clone();
        Mat original = image.clone();
        cvtColor(image, image, COLOR_BGR2HSV);
        Mat lower = Mat(colour[0], true);
        Mat upper = Mat(colour[1], true);
        Mat mask;
        inRange(image, lower, upper, mask);
        Mat detected;
        bitwise_and(original, original, detected, mask = mask);
        bitwise_or(final_image, detected, final_image);
    }

    return final_image;
}

vector<vector<Point>> setup_contours(Mat img, double epsilon = 20.0) {
    Mat cannied;
    Canny(img, cannied, 200, 600);
    vector<vector<Point>> contours0;
    findContours(cannied.clone(), contours0, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> contours;
    contours.reserve(contours0.size());

    for (auto& cnt : contours0) {
        vector<Point> approx;
        approxPolyDP(cnt, approx, epsilon, true);
        contours.push_back(approx);
    }

    return contours;
}

Mat produce_contours(Mat img, double epsilon = 10.0) {
    vector<vector<Point>> contours = setup_contours(img, epsilon);
    Mat vis = Mat::zeros(img.size(), CV_8UC3);

    drawContours(vis, contours, -1, Scalar(255, 255, 255), 3, LINE_AA);
    return vis;
}

void produce_individual_contours(Mat img) {
    vector<vector<Point>> contours = setup_contours(img);

    Mat vis = Mat::zeros(img.size(), CV_8UC3);

    for (size_t i = 0; i < min((size_t) 3, contours.size()); ++i) {
        vector<vector<Point>> single_contour = {contours[i]};
        drawContours(vis.clone(), single_contour, 0, Scalar(255, 255, 255), 3, LINE_4);
        imshow("contour" + to_string(i), vis);
    }
}

Mat fill_image(Mat img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    threshold(gray, gray, 0, 240, THRESH_BINARY);
    return gray;
}

Mat connect(Mat img) {
    Mat labels, stats, centroids;
    int nlabels = connectedComponentsWithStats(img, labels, stats, centroids, 8, CV_32S);
    vector<int> areas(stats.ptr<int>(), stats.ptr<int>() + nlabels);
    Mat result = Mat::zeros(labels.size(), CV_8U);

    for (int i = 1; i < nlabels; ++i) {
        if (areas[i] >= 5000) {
            result.setTo(255, labels == i);
        }
    }

    return result;
}

double distance(Point2d p1, Point2d p2) {
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

void detect_lines(Mat img, vector<Vec4f>& best_lines, Mat& with_lines) {
    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_STD);
    vector<Vec4f> lines;
    lsd->detect(img, lines);
    
    sort(lines.begin(), lines.end(), [&img](const Vec4f& a, const Vec4f& b) {
        return distance(Point2d(a[0], a[1]), Point2d(a[2], a[3])) > distance(Point2d(b[0], b[1]), Point2d(b[2], b[3]));
    });

    best_lines.assign(lines.begin(), lines.begin() + min(5, static_cast<int>(lines.size())));
    with_lines = Mat::zeros(img.size(), CV_8UC3);
    lsd->drawSegments(with_lines, best_lines);
}

vector<Point2d> compute_points(vector<Point2d> orig_points) {
    vector<Point2d> points = orig_points;
    Point2d a = orig_points[0];
    Point2d b = orig_points[1];
    Point2d c = orig_points[2];
    Point2d d = orig_points[3];

    Point2d e = (c + a - b) + (c - b) * 0.02 + (a - b) * 0.1;
    points.push_back(e);

    Point2d dr = c - b;
    Point2d dc = a - b;

    vector<double> coeffs = {1.0 / 6, 0.5, 5.0 / 6};

    for (double m1 : coeffs) {
        for (double m2 : coeffs) {
            points.push_back(b + dr * m1 + dc * m2);
        }
    }

    dr = d - c;
    dc = e - c;

    for (double m1 : coeffs) {
        for (double m2 : coeffs) {
            points.push_back(c + dr * m1 + dc * m2);
        }
    }

    return points;
}

Mat plot_points(Mat img, const vector<Point2d>& points) {
    for (const Point2d& pt : points) {
        circle(img, pt, 1, Scalar(255, 255, 255), 2);
    }
    return img;
}

void produce_image() {
    Mat img = get_image();
    img = scale_image(img);
    imshow("original", img);

    Mat mask_img = mask_image(img);
    // imshow("masked", mask_img);

    Mat blurred;
    GaussianBlur(mask_img, blurred, Size(5, 5), 0);
    Mat filled = fill_image(blurred);
    // imshow("filled", filled);

    Mat contours = produce_contours(img);
    // imshow("contours", contours);

    // produce_individual_contours(img);

    Mat connected_comps = connect(filled);
    // imshow("connected", connected_comps);

    Mat again;
    GaussianBlur(connected_comps, again, Size(3, 3), 0);
    again = produce_contours(again, 20.0);
    // imshow("contours second", again);

    Mat combined = again | connected_comps;
    // cout << combined << endl;

    vector<Vec4f> best_lines;
    Mat with_lines;
    GaussianBlur(combined, with_lines, Size(77, 77), 0);
    detect_lines(with_lines, best_lines, with_lines);
    // detect_lines(combined, best_lines, with_lines);
    // detect_lines(again, best_lines, with_lines);
    // imshow("with lines", with_lines);

    Mat two_lines = with_lines;
    vector<Point2d> points;
    Mat image_with_points = plot_points(mask_img, compute_points(points));
    imshow("points image", image_with_points);
}

int main(int argc, char** argv) {
    if (USE_CAM) {
        cap.open(0);
        if (!cap.isOpened()) {
            cerr << "Camera not initialized." << endl;
            return -1;
        }
    }

    int waitTime = 330;

    if (USE_CAM) {
        while (true) {
            produce_image();
            char key = static_cast<char>(waitKey(waitTime) & 0xFF);
            if (key == 'q') {
                break;
            }
        }
    } else {
        produce_image();
    }

    waitKey(0);
    destroyAllWindows();

    return 0;
}
