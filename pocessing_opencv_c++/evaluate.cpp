#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <cmath>
#include <cstdio>

#include "evaluate.h"

using namespace cv;
using namespace std;


float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for (int i = 0; i < actual.rows; i++) {
        float p = predicted.at<float>(i, 0);
        float a = actual.at<int>(i, 0);
        //cout << "actual: " << a << "\t" << "predicted: "<< p << endl;
        //if ((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
        if (p == a) {
        //if ((int) predicted.at<float>(i) == actual.at<int>(i)){
    
            t++;
        }
        else {
            f++;
        }
   //cout << "--*-*p-*--" << p << "---a----" << a << endl; 

    }
    ///cout << "--*-*T-*--" << t << "---F----" << f << endl;
    return (t * 1.0) / (t + f);
}


/*
float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        int p = predicted.at<float>(i,0);
        float a = actual.at<float>(i,0);
        if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
            t++;
        } else {
            f++;
        }
    }
    return (t * 1.0) / (t + f);
}
*/
/*
if((p >= 0.0 && a >= 0.0) || (p <= 0.0 &&  a <= 0.0)) {
      t++;
    } else {
      f++;
    }
*/