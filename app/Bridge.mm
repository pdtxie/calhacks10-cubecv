#import <opencv2/opencv.hpp>
#import <opencv2/imgcodecs/ios.h>
#import <Foundation/Foundation.h>
#import "Bridge.h"
#include "identify.hpp"

@implementation Bridge

- (int*) cubecv: (UIImage *) img {
    cv::Mat opencvImage;
    UIImageToMat(img, opencvImage, true);
    
//    cv::Mat convertedColorSpaceImage;
//    cv::cvtColor(opencvImage, convertedColorSpaceImage, CV_RGBA2RGB);
    
    return cubecv(opencvImage);
}


@end
