#include "identify.hpp"

using namespace cv;
using namespace std;

bool USE_CAM = true;
VideoCapture cap(1);

enum Colour {
    BLUE, GREEN, ORANGE, RED, YELLOW
};


map<Colour, vector<vector<int>>> RANGES;
vector<vector<Colour>> piece_colours;
int faces[54];


void print_array(int* v) {
    for (int i = 0; i < 54; i++) {
        cout << v[i] << " ";
    }
    cout << endl;
}



Mat get_image() {
    Mat img;

    if (USE_CAM) {
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

    for (auto it = RANGES.begin(); it != RANGES.end(); it++) {
        Mat image = img.clone();
        Mat original = image.clone();
        cvtColor(image, image, COLOR_BGR2HSV);
        Mat lower = Mat((*it).second[0], true);
        Mat upper = Mat((*it).second[1], true);

        Mat mask;
        inRange(image, lower, upper, mask);

        Mat detected;
        bitwise_and(original, original, detected, mask = mask);
        bitwise_or(final_image, detected, final_image);
    }

    return final_image;
}

vector<vector<Point>> setup_contours(Mat img, double epsilon) {
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
    Mat vis = Mat::zeros(img.size(), CV_8U);

    drawContours(vis, contours, -1, Scalar(255, 255, 255), 3, LINE_AA);
    return vis;
}

void produce_individual_contours(Mat img) {
    vector<vector<Point>> contours = setup_contours(img, 20.0);

    Mat vis = Mat::zeros(img.size(), CV_8UC3);

    for (size_t i = 0; i < min((size_t) 3, contours.size()); ++i) {
        vector<vector<Point>> single_contour = {contours[i]};
        drawContours(vis.clone(), single_contour, 0, Scalar(255, 255, 255), 3, LINE_4);
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
    int nlabels = connectedComponentsWithStats(img, labels, stats, centroids);

    Mat result = Mat::zeros(labels.size(), CV_8U);

    for (int i = 1; i < stats.rows; i++) {
        if (stats.at<int>(i, 4) >= 5000) {
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
    // with_lines = Mat::zeros(img.size(), CV_8UC3);
    
    if (best_lines.size() > 0)
        lsd->drawSegments(with_lines, best_lines);
}


vector<Point> find_points(vector<Vec4f> lines, Mat &img) {
    Vec4f topmost = lines[0];
    Vec4f leftmost = lines[0];

    for (const Vec4f& line : lines) {
        int max_x = max(line[0], line[2]);
        int max_y = max(line[1], line[3]);

        if (max_x < max(topmost[0], topmost[2]))
            topmost = line;

        if (max_y < max(leftmost[1], leftmost[3]))
            leftmost = line;
    }

    if (topmost[0] > topmost[2]) {
        swap(topmost[0], topmost[2]);
        swap(topmost[1], topmost[3]);
    }

    if (leftmost[1] < leftmost[3]) {
        swap(leftmost[0], leftmost[2]);
        swap(leftmost[1], leftmost[3]);
    }

    Point a(topmost[2], topmost[3]); // top right - red
    Point b(topmost[0], topmost[1]); // top left - blue
    Point c(leftmost[2], leftmost[3]); // left top - yellow
    Point d(leftmost[0], leftmost[1]); // left bottom - green

    // circle(img, a, 1, Scalar(0, 0, 255), 2);
    // circle(img, b, 1, Scalar(255, 0, 0), 2);
    // circle(img, c, 1, Scalar(0, 255, 255), 2);
    // circle(img, d, 1, Scalar(0, 255, 0), 2);

    return vector<Point>{d, c, b, a};
}

void rotate_n(vector<vector<int>> &face, int n) {
    for (int i = 0; i < n; i++) {
        swap(face[0][0], face[2][2]);
        swap(face[0][1], face[1][2]);
        swap(face[1][0], face[2][1]);
        swap(face[0], face[2]);
    }
}



vector<Point> compute_points(vector<Point> orig_points) {
    // vector<Point> points = orig_points;
    vector<Point> points;

    Point a = orig_points[0];
    Point b = orig_points[1];
    Point c = orig_points[2];
    Point d = orig_points[3];

    Point e = (c + a - b) + (c - b) * 0.02 + (a - b) * 0.1;
    // points.push_back(e);

    Point dr = c - b;
    Point dc = a - b;

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

Mat plot_points(Mat img, const vector<Point>& points) {
    for (const Point& pt : points) {
        circle(img, pt, 1, Scalar(255, 255, 255), 2);
    }

    return img;
}


optional<Colour> get_colour(Vec3b c) {
    auto it = RANGES.begin();

    while (it != RANGES.end()) {
        vector<vector<int>> value_range = (*it).second;
        
        if (c[2] == 0) return nullopt;

        if (value_range[0][0] <= c[0] && value_range[1][0] >= c[0]) {
            return make_optional((*it).first);
        }

        it++;
    }

    return nullopt;  // WARN: should never happen
}




int* produce_image(Mat img) {
//    Mat img = get_image();
    img = scale_image(img);

    Mat mask_img = mask_image(img);

    Mat blurred;
    GaussianBlur(mask_img, blurred, Size(5, 5), 0);
    Mat filled = fill_image(blurred);
    // imshow("filled", filled);

    Mat contours = produce_contours(img);
    // imshow("contours", contours);

    Mat connected_comps = connect(filled);
    // imshow("connected", connected_comps);

    Mat again;
    GaussianBlur(connected_comps, again, Size(3, 3), 0);
    again = produce_contours(again, 20.0);

    
    Mat combined;
    bitwise_or(again, connected_comps, combined);
    // imshow("combined", combined);

    vector<Vec4f> best_lines;
    Mat with_lines;
    GaussianBlur(combined, with_lines, Size(77, 77), 0);
    detect_lines(with_lines, best_lines, with_lines);

    if (best_lines.size() <= 2) return faces;

    vector<Point> points = find_points(best_lines, with_lines);

    Mat two_lines = with_lines;
    auto piece_points = compute_points(points);

    Mat image_with_points = plot_points(mask_img.clone(), piece_points);




    Mat hsb_final;
    cvtColor(mask_img, hsb_final, COLOR_BGR2HSV);


    int color_top = 5, color_bottom = 5;
    vector<vector<int>> bottom_face = vector<vector<int>>(3, vector<int>(3, 5));

    for (int i = 0; i < piece_points.size(); i++) {
        Vec3b hsv = hsb_final.at<Vec3b>(piece_points[i].y, piece_points[i].x);


        auto x = get_colour(hsb_final.at<Vec3b>(piece_points[i].y, piece_points[i].x));

        if (x) {
            cout << x.value() << endl;
        } else {
            cout << 5 << endl;
            continue;
        }

        if (i >= 9) {
            bottom_face[i / 3 - 3][i % 3] = x.value();
        }

        if (i == 4) {
            color_top = x.value();
        }
        else if (i == 13) {
            color_bottom = x.value();
        }
    }

    int code = color_top * 10 + color_bottom;

    if (code % 11 == 0 || code == 10 || code == 1 || code == 23 || code == 32 || code == 45 || code == 54) {
        return faces;
    }

    if (code == 35 || code == 34 || code == 31 || code == 3 || code == 20 || code == 12) {
        rotate_n(bottom_face, 1);
    }
    else if (code == 15 || code/10 == 4 || code == 4) {
        rotate_n(bottom_face, 2);
    }
    else if (code == 25 || code == 24 || code == 13 || code == 30 || code == 2 || code == 21) {
        rotate_n(bottom_face, 3);
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == 1 && j == 1) {
                faces[color_bottom * 9 + 4] = color_bottom;
            }
            
            faces[color_bottom * 9 + 3 * i + j] = bottom_face[i][j];
        }
    }

    
    return faces;
}




int* cubecv(Mat img) {
    RANGES[BLUE] = {{100, 143, 145}, {118, 255, 255}};
    RANGES[GREEN] = {{43, 153, 0}, {132, 255, 255}};
    RANGES[ORANGE] = {{0, 172, 83}, {20, 255, 255}};
    RANGES[RED] = {{170, 145, 80}, {179, 255, 255}};
    RANGES[YELLOW] = {{28, 145, 0}, {40, 255, 255}};

//    if (USE_CAM) {
//        cap.open(1);
//    }
//
//    int waitTime = 330;
//
//    if (USE_CAM) {
//        while (true) {
    
    Mat newimg;
    cvtColor(img, newimg, COLOR_RGBA2BGR);
    
    return produce_image(newimg);
    
//            char key = static_cast<char>(waitKey(waitTime) & 0xFF);
//            if (key == 'q') {
//                break;
//            }
//        }
//    } else {
//        produce_image(img);
//    }
//
//    waitKey(0);
//    destroyAllWindows();
}
