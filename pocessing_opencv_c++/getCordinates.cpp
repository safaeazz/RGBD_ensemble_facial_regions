#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <cmath>
#include <cstdio>

using namespace cv;
using namespace std;

cv::Mat getCordinates(cv::Rect& R01, cv::Mat& depth)
{

  return depth(R01);
}
