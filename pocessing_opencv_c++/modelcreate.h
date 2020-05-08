#ifndef _MODELCREATE_
#define _MODELCREATE_
#include <iostream>
#include "PAuto.h"

void modelcreate(std::string path,std::string type,PAuto feature,std::string name,cv::Mat& telabels,cv::Mat& predicted,float& score1);
//void svm(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses);
//void modelcreate(std::string path,std::string type,std::string name,cv::Mat& predicted,float& score1);

#endif