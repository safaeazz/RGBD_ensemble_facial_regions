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

void gabormatdepth(cv::Mat& image,cv::Mat& vec)
{

    Mat src,dest; 

       if(image.type()!=0){
    cvtColor( image,image, COLOR_BGR2GRAY );
    //equalizeHist( img,img );
}                                   

    //image.convertTo(image,CV_32FC1);
    //int pos_sigma = 3.00, pos_th =116, pos_lm = 36.00, pos_psi = 274.00;//5,2,50,90
    //int pos_sigma = 2.00, pos_th =22, pos_lm = 50.00, pos_psi = 90.00;//5,2,50,90

    int kernel_size = 21;//21
    int pos_sigma = 5.00, pos_th =0, pos_lm =42.00, pos_psi = 95.00;//5,2,50,90
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
}
