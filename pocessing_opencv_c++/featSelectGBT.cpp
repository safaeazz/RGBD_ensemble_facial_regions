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


using namespace std;
using namespace cv;



cv::Mat featSelectGBT(cv::Mat& training_data, cv::Mat& training_classifications){

    int NUMBER_OF_TRAINING_SAMPLES= training_data.size().height;
    int ATTRIBUTES_PER_SAMPLE=training_data.size().width;
    int NUMBER_OF_CLASSES=3;
    // load training and testing data sets


    Mat new_data = Mat(NUMBER_OF_TRAINING_SAMPLES*NUMBER_OF_CLASSES, ATTRIBUTES_PER_SAMPLE + 1, CV_32F );
    Mat new_responses = Mat(NUMBER_OF_TRAINING_SAMPLES*NUMBER_OF_CLASSES, 1, CV_32S );

        // 1. unroll the training samples



        printf( "\nUnrolling the database...");
        for(int i = 0; i < NUMBER_OF_TRAINING_SAMPLES; i++ )
        {
            for(int j = 0; j < NUMBER_OF_CLASSES; j++ )
            {
                for(int k = 0; k < ATTRIBUTES_PER_SAMPLE; k++ )
                {

                    new_data.at<float>((i * NUMBER_OF_CLASSES) + j, k) = training_data.at<float>(i, k);

                }

                new_data.at<float>((i * NUMBER_OF_CLASSES) + j, ATTRIBUTES_PER_SAMPLE) = (float) j;
                if ( ( (int) training_classifications.at<float>( i, 0)) == j)
                {
                    new_responses.at<int>((i * NUMBER_OF_CLASSES) + j, 0) = 1;
                }
                else
                {
                    new_responses.at<int>((i * NUMBER_OF_CLASSES) + j, 0) = 0;
                }
            }
        }
        printf( "Done\n");

        Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 2, 1, CV_8U );
        var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
        var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;
        var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE + 1, 0) = CV_VAR_CATEGORICAL;


        float priors[] = {(NUMBER_OF_CLASSES-1),1};
       // set the boost parameters
      CvBoostParams params = CvBoostParams(CvBoost::REAL,  // boosting type
                           100,      // number of weak classifiers
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
      cout << "--newrespiinse--"  << new_responses<< endl;
      boost->train( new_data, CV_ROW_SAMPLE, new_responses, Mat(), Mat(), var_type,
                              Mat(), params, false);

      printf( "----Done.");

      Mat idx;
      std::vector<int> featureIndexes; 
      CvBoostTree * predictor;
      CvSeq *predictors = boost->get_weak_predictors();
      
      if (predictors){
       
         //cout<<" Number of weak predictors="<<predictors->total<<endl;

         for (int i = 0; i < predictors->total; i++) {
           predictor = *CV_SEQ_ELEM(predictors, CvBoostTree*, i);
           const CvDTreeNode * node = predictor->get_root();          
            CvDTreeSplit* split = node->split;
            const int index = split->var_idx;
           cout<<" index="<<i<<"----"<< split->var_idx<<endl;


         }
         
      }

      cv::sort(idx, idx, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);

      return idx;






/******************************************************************************/

} 