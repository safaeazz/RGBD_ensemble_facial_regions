#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>

#include "svm.h"
#include "evaluate.h"
#include "RTree.h"


using namespace std;
using namespace cv;

//#define ATTRIBUTES_PER_SAMPLE 1152000
// cv::Mat& testdata, cv::Mat& testClasses

//void RTree(cv::Mat& trainingdata, cv::Mat& trainingClasses)
void RTree(cv::Mat& trainingdata, cv::Mat& trainingClasses,string name, cv::Mat& testdata, cv::Mat& testClasses,cv::Mat& predicted,float& score)

//,cv::Mat& testdata, cv::Mat& testClasses)
{
  
  cv::Mat var_type = cv::Mat(trainingdata.size().width + 1, 1, CV_8UC1);
  var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
  var_type.at<uchar>(trainingdata.size().width, 0) = CV_VAR_CATEGORICAL; //CV_VAR_ORDERED

  // this is a classification problem (i.e. predict a discrete number of class
  // outputs) so reset thvar_typee last (+1) output var_type element to CV_VAR_CATEGORICAL

  float priors[] = { 1, 1,1};  // weights of each classification for classes
  // (all equal as equal samples of each digit)

  CvRTParams paramms = CvRTParams(25, // max depth
    5, // min sample count
    0, // regression accuracy: N/A here
    false, // compute surrogate split, no missing data
    15, // max number of categories (use sub-optimal algorithm for larger numbers)
    priors, // the array of priors
    false,  // calculate variable importance
    4,       // number of variables randomly selected at node and used to find the best split(s).
    1000,   // max number of trees in the forest
    0.01f,             // forrest accuracy
    CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
    );
  CvRTrees* rtree = new CvRTrees;

  rtree->train(trainingdata, CV_ROW_SAMPLE, trainingClasses, cv::Mat(), cv::Mat(), var_type, cv::Mat(), paramms);
 // rtree->save("rtree.xml");

  /*float acc;
  acc = rtree->get_train_error();*/
  //rtree->save("./Classifieur/RandomForest/LPB/Classification_Depth_1000trees.xml");
  //rtree->load("./Classifieur/RandomForest/LPB/Classification_Depth_1000.xml");
  //rtree->load("./Classes/RTREE_RGB_LBP_VECTEURSIZE15.xml");

 //cv::Mat predicted(testClasses.rows, 1, CV_32F);
 //c::Mat predicted(testClasses.rows, 1, CV_32F);
    predicted= Mat(testClasses.rows, 1,CV_32F);

  for (int i = 0; i < testdata.rows; i++) {
    const cv::Mat sample = testdata.row(i);

    predicted.at<float>(i, 0) = rtree->predict(sample);
  }
  score=evaluate(predicted,testClasses);


  //cout << "Accuracy_{TREE} = " << evaluate(predicted, testClasses) << endl;

  //result = rtree->predict(testdata, Mat());



  //cout << "Accuracy_{Random forest} = " << evaluate(res1, labels) << endl;
  // trainnig data and labels size
  //cout << "training data Mat :" << trainingdata.size() << endl;
  //cout << "test data Mat :" << testdata.size() << endl;
  //cout << "labels data Mat :" << labels.size() << endl;
}