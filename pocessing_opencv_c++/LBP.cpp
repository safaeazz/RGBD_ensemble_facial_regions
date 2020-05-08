#include <iostream>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/features2d.hpp> 
#include "opencv2/opencv.hpp"
#include "LBP.h"
#include "histImage.h"
#include <limits>


using namespace std;
using namespace cv;



void LBP(Mat& img,cv::Mat& vec1){


    Mat dst = Mat::zeros(img.rows-2, img.cols-2, CV_8UC1);//CV_8UC1
   
    for(int i=1;i<img.rows-1;i++) {
        for(int j=1;j<img.cols-1;j++) {
            uchar center = img.at<uchar>(i,j);
            unsigned char code = 0;
            code |= ((img.at<uchar>(i-1,j-1)) > center) << 7;
            code |= ((img.at<uchar>(i-1,j)) > center) << 6;
            code |= ((img.at<uchar>(i-1,j+1)) > center) << 5;
            code |= ((img.at<uchar>(i,j+1)) > center) << 4;
            code |= ((img.at<uchar>(i+1,j+1)) > center) << 3;
            code |= ((img.at<uchar>(i+1,j)) > center) << 2;
            code |= ((img.at<uchar>(i+1,j-1)) > center) << 1;
            code |= ((img.at<uchar>(i,j-1)) > center) << 0;
            dst.at<uchar>(i-1,j-1) = code;
        }
    }

    //dst.convertTo(dst,CV_8UC1);

      //Mat im1 = histImage(dst);
      //cout << "---" << im1.size() << endl;
      vec1 = dst.reshape(0,1);
      //normalize(vec1,vec1 ,0, 1, NORM_MINMAX, CV_8UC1);

      

}
