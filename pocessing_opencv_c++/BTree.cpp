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
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include "opencv2/core/types_c.h"
#include "opencv2/gpu/gpu.hpp"

#include "svm.h"
#include "evaluate.h"
#include "BTree.h"


using namespace std;
using namespace cv;


void BTree(cv::Mat& trainingdata, cv::Mat& trainingClasses,string name,cv::Mat& testdata, cv::Mat& testClasses,Mat& predicted,float& score)
{

  
  cv::Mat var_type = cv::Mat(trainingdata.size().width + 1, 1, CV_8UC1);
  var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL));
  var_type.at<uchar>(trainingdata.size().width, 0) = CV_VAR_CATEGORICAL; //CV_VAR_ORDERED
  float priors[] = { 1, 1};  // weights of each classification for classes
  CvBoostParams params = CvBoostParams(CvBoost::REAL,  // boosting type
                       1000, // number of weak classifiers
                       0.95, // trim rate
                       1,    // max depth of trees
                       false,  // compute surrogate split, no missing data
                       priors );

  params.max_categories = 15;   // max number of categories (use sub-optimal algorithm for larger numbers)
  params.min_sample_count = 5;  // min sample count
  params.cv_folds = 1;          // cross validation folds
  params.use_1se_rule = false;      // use 1SE rule => smaller tree
  params.truncate_pruned_tree = false;  // throw away the pruned tree branches
  params.regression_accuracy = 0.0;     // regression accuracy: N/A here
  
  CvBoost* boost = new CvBoost;
  boost->train( trainingdata, CV_ROW_SAMPLE, trainingClasses,  Mat(), Mat(), var_type,
                       Mat(), params, false);
  //boost->save("./files/rtree.xml");
  //boost->load("./files/rtree.xml");
string dest;
dest = "./files/boost/"+name+".xml";
boost->save(dest.c_str());

  predicted=Mat(testClasses.rows, 1, CV_32F);
  for (int i = 0; i < testdata.rows; i++) {
  const cv::Mat sample = testdata.row(i);
  predicted.at<float>(i, 0) = boost->predict(sample);
  //cout <<">>> / "<<testClasses.row(i) <<" // "<< boost->predict(sample)<<endl;

  }
    
score = evaluate(predicted,testClasses);
cout << "adaboost score --" << score<< endl;
  //cout << "Accuracy_{SVM} = " << evaluate(predicted, testClasses) << " % " << endl;

 //1. Declare a structure to keep the data

  /****Here ****/
   //CvSeq* weak;
  //weak = boostTree->get_weak_predictors();
  
  /*
   CvSeqReader reader;
   
   cvStartReadSeq( weak, &reader );
   cvSetSeqReaderPos( &reader, 0 );
  
      printf("%d Elements\n",weak->total);
      std::vector<int> featureIndexes;
      int weak_count = weak->total;
*/

    //Only add features that are not already added
    // cout << "data***----**--" << seq->elem_size << endl;
    //   CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
     
  /*
      for( int i = 0; i < weak_count; i++ ) {
          CvBoostTree* wtree;
          CV_READ_SEQ_ELEM( wtree, reader );
          const CvDTreeNode* node = wtree->get_root();
          CvDTreeSplit* split = node->split;
          const int index = split->condensed_idx;
          
  cout << "WC = " <<weak->first->start_index << endl;
   CV_NEXT_SEQ_ELEM( weak->elem_size, reader );
   if (std::find(featureIndexes.begin(),
                 featureIndexes.end(),
                 index) == featureIndexes.end()) {
     cout << weak->elem_size << endl;
     featureIndexes.push_back(index);
   }
 }   
 */ 
      //Only add features that are not already added
 /*
          if (std::find(featureIndexes.begin(),
                        featureIndexes.end(),
                        index) == featureIndexes.end()) {
             cout << "I===" << index << endl;

              featureIndexes.push_back(index);
          }

      }
      */
   // Mat idx = Mat(featureIndexes,CV_32S);
        // printf("%d\n", featureIndexes);
    //cout << "index ===" << idx.size() <<endl;





  //boostTree->save("./files/Btree.xml");


//boostTree->calc_error(trainingdata,CV_TRAIN_ERROR, trainingClasses);
//boostTree->calc_error(trainingdata,trainingClasses);
//cout << "error====" << fl1 << endl;
}