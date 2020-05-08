#ifndef _RTREE_
#define _RTREE_
#include <iostream>


//void RTree(cv::Mat& trainingData, cv::Mat& trainingClasses);
void RTree(cv::Mat& trainingdata, cv::Mat& trainingClasses,std::string name, cv::Mat& testdata, cv::Mat& testClasses,cv::Mat& predicted,float& score);

//,cv::Mat& testData, cv::Mat& testClasses);
// 
#endif