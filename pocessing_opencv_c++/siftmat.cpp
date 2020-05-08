#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"


#include "siftmat.h"

using namespace cv;
using namespace std;

void siftmat(cv::Mat& input, Mat& vec1)
{

   cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(input, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    // imshow("",output);waitKey();
     Mat vectx = Mat(keypoints,CV_8UC4);
        //  cout << "key === " << vectx.size()<< endl;

    vec1 =output.reshape(0,1);


/*
  SIFT sift;
  vector<KeyPoint> keypoints;
  sift(image, noArray(), keypoints, noArray());
  Mat output1;
  drawKeypoints(image, keypoints, output1);
  //Mat
   //vec1 = Mat(keypoints);
   //cout << "-------**----" << vectx.size() << endl;
   imshow("",output1);waitKey();
  vec1 = output1.reshape(0,1);
 //cout << "-------**---********-" << vec1.size() << endl;
 */
}