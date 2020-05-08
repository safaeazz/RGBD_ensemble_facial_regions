#ifndef _GBT_
#define _GBT_
#include <iostream>

void GBT(cv::Mat& trainingdata, cv::Mat& trainingClasses,std::string name, cv::Mat& testdata, cv::Mat& testClasses,cv::Mat& predicted,float& score);
//void GBT(cv::Mat& trainingData, cv::Mat& trainingClasses,cv::Mat& testData, cv::Mat& testClasses);
// cv::Mat& testData, cv::Mat& testClasses
#endif