#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cstdlib>
#include <ctime>


/** OPENCV **/

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"


using namespace std;
using namespace cv;


// get the indices of selected features

cv::Mat featSelectBoost(cv::Mat& trainingdata, cv::Mat& trainingClasses){

  cv::Mat var_type = cv::Mat(trainingdata.size().width + 1, 1, CV_8UC1);
  var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL));
  var_type.at<uchar>(trainingdata.size().width, 0) = CV_VAR_CATEGORICAL; //CV_VAR_ORDERED


  float priors[] = { 1,1};  // weights of each classification for classes

  CvBoostParams params = CvBoostParams(CvBoost::REAL,  // boosting type
                       1000,      // number of weak classifiers
                       0.95,       // trim rate
                       1,    // max depth of trees
                       false,  // compute surrogate split, no missing data
                       priors );

  params.max_categories = 15;   // max number of categories (use sub-optimal algorithm for larger numbers)
  params.min_sample_count = 5;  // min sample count
  params.cv_folds = 0;          // cross validation folds
  params.use_1se_rule = false;      // use 1SE rule => smaller tree
  params.truncate_pruned_tree = false;  // throw away the pruned tree branches
  params.regression_accuracy = 0.0;     // regression accuracy: N/A here
  
  CvBoost* boost = new CvBoost;
  boost->train( trainingdata, CV_ROW_SAMPLE, trainingClasses,  Mat(), Mat(), var_type,
                     Mat(), params, false);
 //boost->load("./results/Gender2/adaboostmodel/hogRGB.xml");
// boost->save("./results/hogdepth.xml");
  Mat idx;
  std::vector<int> featureIndexes;
  //set<unsigned int> * selected = new set<unsigned int>();  
  CvBoostTree * predictor;
  CvSeq *predictors = boost->get_weak_predictors();
  
  if (predictors){

     for (int i = 0; i < predictors->total; i++) {
       predictor = *CV_SEQ_ELEM(predictors, CvBoostTree*, i);
       const CvDTreeNode * node = predictor->get_root();
       
        CvDTreeSplit* split = node->split;
        const int index = split->var_idx;
 
        if (std::find(featureIndexes.begin(),
                      featureIndexes.end(),
                      index) == featureIndexes.end()) {

            featureIndexes.push_back(index);
        }

       const CvMat * var_importance = predictor->get_var_importance();  // <--- This seems to make memory grow..
       
       for (int j = 0; j < var_importance->cols * var_importance->rows; j++) {
         double val = var_importance->data.db[j];
         if (val > 0) {

           idx.push_back(j);
          
         }
       }
     
     }
     
  }
  Mat feidx = Mat(featureIndexes,CV_32S);
  cv::sort(feidx, feidx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
  cv::sort(idx, idx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);

  //cout << "feature size========" << idx << endl;
  //cout << "feature size******sirted***" << idx << endl;
  return idx;
}  

//cout << "vars" << idx << endl;