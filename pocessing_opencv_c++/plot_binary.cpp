#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <cmath>
#include <cstdio>

using namespace cv;
using namespace std;



void plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
int size=200;
  cv::Mat plot(size, size, CV_8UC3);
  plot.setTo(cv::Scalar(255.0,255.0,255.0));
  for(int i = 0; i < data.rows; i++) {

    float x = data.at<float>(i,0) * size;
    float y = data.at<float>(i,1) * size;

    if(classes.at<float>(i, 0) > 0) {
      cv::circle(plot, Point(x,y), 2, CV_RGB(255,0,0),1);
    } else {
      cv::circle(plot, Point(x,y), 2, CV_RGB(0,255,0),1);
    }
  }
  cv::imshow(name, plot);
}