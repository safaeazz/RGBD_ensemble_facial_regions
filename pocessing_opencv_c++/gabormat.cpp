#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <cmath>
#include <cstdio>

#include "mkKernel.h"


using namespace cv;
using namespace std;

void gabormat(cv::Mat& image,cv::Mat& vec)
{

      Mat src;
      Mat dest; 

    //image.convertTo(image,CV_32FC1);
    //int pos_sigma = 5.00, pos_th =6, pos_lm = 50.00, pos_psi = 90.00;//5,2,50,90
    //int pos_sigma = 3.00, pos_th =116, pos_lm = 36.00, pos_psi = 274.00;//5,2,50,90

    //sigma=3 lambda=36 theta=116 psi=274
    //Mat vec;sigma=3 lambda=36 theta=116 psi=274
    
    int kernel_size = 21;//21
    int pos_sigma = 4.00, pos_th =116, pos_lm = 36.00, pos_psi = 90.00;//90
    double sig = pos_sigma;
    double lm = 0.5+pos_lm/100.0;
    double th = pos_th;
    double ps = pos_psi;
    
    image.convertTo(image,CV_32FC1, 1.0/255, 0);
    cv::Mat kernel = mkKernel(kernel_size, sig, th, lm, ps);
    cv::filter2D(image, dest, CV_32F, kernel);
    cv::Mat mag;
    cv::pow(dest, 2.0, mag);
    vec = mag.reshape(0,1);
   // imshow("",mag);waitKey();
     //return vec; 
      

}