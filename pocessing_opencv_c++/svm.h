#ifndef _SVM_
#define _SVM_
#include <iostream>

//void svm(cv::Mat& trainingData, cv::Mat& trainingClasses,std::string name);
void svm(cv::Mat& trainingData, cv::Mat& trainingClasses,std::string name, cv::Mat& testData, cv::Mat& testClasses,cv::Mat& predicted,float& score);
//void svm(cv::Mat& trainingData, cv::Mat& trainingClasses,std::string name,cv::Mat& predicted,float& score);

#endif