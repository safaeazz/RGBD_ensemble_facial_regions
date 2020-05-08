// This is start of the header guard.  ADD_H can be any unique name.  By convention, we use the name of the header file.
#ifndef TDLBP_H
#define TDLBP_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/detection_based_tracker.hpp>
//#include <opencv2/objdetect/objdetect.hpp>

#include <vector>

using namespace cv;
using namespace std;

int dec2bin(int number);
void histogramCount(Mat& Image, vector<int>& histogram, int x1, int x2, int y1, int y2);
int bin2dec(int number);
Mat calculate3DLBP(Mat depth_image);

#endif