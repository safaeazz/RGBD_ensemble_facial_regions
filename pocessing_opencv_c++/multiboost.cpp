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

#include "evaluate.h"


using namespace std;
using namespace cv;


void multiboost(cv::Mat& training_data, cv::Mat& training_classifications,cv::Mat& testing_data, cv::Mat& testing_classifications)
{


 int NUMBER_OF_TRAINING_SAMPLES= training_data.size().height;
 int ATTRIBUTES_PER_SAMPLE=training_data.size().width;
 int NUMBER_OF_TESTING_SAMPLES=testing_data.size().height;
 int NUMBER_OF_CLASSES=3;

  Mat new_data = Mat(NUMBER_OF_TRAINING_SAMPLES*NUMBER_OF_CLASSES, ATTRIBUTES_PER_SAMPLE + 1, CV_32F );
  Mat new_responses = Mat(NUMBER_OF_TRAINING_SAMPLES*NUMBER_OF_CLASSES, 1, CV_32S );

     // 1. unroll the training samples

        printf( "\nUnrolling the database...");
        fflush(NULL);
        for(int i = 0; i < NUMBER_OF_TRAINING_SAMPLES; i++ )
        {
            for(int j = 0; j < NUMBER_OF_CLASSES; j++ )
            {
                for(int k = 0; k < ATTRIBUTES_PER_SAMPLE; k++ )
                {

                    // copy over the attribute data

                    new_data.at<float>((i * NUMBER_OF_CLASSES) + j, k) = training_data.at<float>(i, k);

                }

                // set the new attribute to the original class

                new_data.at<float>((i * NUMBER_OF_CLASSES) + j, ATTRIBUTES_PER_SAMPLE) = (float) j;

                // set the new binary class

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

        // 2. Unroll the type mask

        // define all the attributes as numerical
        // alternatives are CV_VAR_CATEGORICAL or CV_VAR_ORDERED(=CV_VAR_NUMERICAL)
        // that can be assigned on a per attribute basis

        Mat var_type = Mat(ATTRIBUTES_PER_SAMPLE + 2, 1, CV_8U );
        var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical

        // this is a classification problem (i.e. predict a discrete number of class
        // outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL
        // *** the last (new) class indicator attribute, as well
        // *** as the new (binary) response (class) are categorical

        var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE, 0) = CV_VAR_CATEGORICAL;
        var_type.at<uchar>(ATTRIBUTES_PER_SAMPLE + 1, 0) = CV_VAR_CATEGORICAL;

        // define the parameters for training the boosted trees

        // weights of each classification for classes
        // N.B. in the "unrolled" data we have an imbalance in the training examples

        float priors[] = {( 3 - 1),1};
        //float priors[] = {1,1};

        // set the boost parameters

        CvBoostParams params = CvBoostParams(CvBoost::REAL,  // boosting type
                                             100,      // number of weak classifiers
                                             0.95,       // trim rate

                                             // trim rate is a threshold (0->1)
                                             // used to eliminate samples with
                                             // boosting weight < 1.0 - (trim rate)
                                             // from the next round of boosting
                                             // Used for computational saving only.

                                             25,    // max depth of trees
                                             false,  // compute surrogate split, no missing data
                                             priors );

        // as CvBoostParams inherits from CvDTreeParams we can also set generic
        // parameters of decision trees too (otherwise they use the defaults)

        params.max_categories = 15;   // max number of categories (use sub-optimal algorithm for larger numbers)
        params.min_sample_count = 5;  // min sample count
        params.cv_folds = 1;          // cross validation folds
        params.use_1se_rule = false;      // use 1SE rule => smaller tree
        params.truncate_pruned_tree = false;  // throw away the pruned tree branches
        params.regression_accuracy = 0.0;     // regression accuracy: N/A here


        // train boosted tree classifier (using training data)

       

        CvBoost* boostTree = new CvBoost;

        boostTree->train( new_data, CV_ROW_SAMPLE, new_responses, Mat(), Mat(), var_type,
                          Mat(), params, false);
        printf( "Done.");

        // perform classifier testing and report results

        Mat test_sample,idx;
        int correct_class = 0;
        int wrong_class = 0;
        int false_positives [3] = {0,0,0};
        Mat weak_responses = Mat( 1, boostTree->get_weak_predictors()->total, CV_32F );
         cout << "--*-*-*-*-"<< boostTree->get_weak_predictors()->total<< endl;
        Mat new_sample = Mat( 1,  ATTRIBUTES_PER_SAMPLE + 1, CV_32F );
        int best_class = 0; // best class returned by weak classifier
        double max_sum;  // highest score for a given class

 

     
/*

        for (int tsample = 0; tsample < NUMBER_OF_TESTING_SAMPLES; tsample++)
        {

            test_sample = testing_data.row(tsample);

            for(int k = 0; k < ATTRIBUTES_PER_SAMPLE; k++ )
            {
                new_sample.at<float>( 0, k) = test_sample.at<float>(0, k);
            }

            // run boosted tree prediction (for N classes and take the
            // maximal response of all the weak classifiers)

            max_sum = INT_MIN; // maximum starts off as Min. Int.

            for(int c = 0; c < NUMBER_OF_CLASSES; c++ )
            {

                new_sample.at<float>(0, ATTRIBUTES_PER_SAMPLE) = (float) c;

                boostTree->predict((new_sample), NULL, (weak_responses));

                Scalar responseSum = sum( weak_responses );

                if( responseSum.val[0] > max_sum)
                {
                    max_sum = (double) responseSum.val[0];
                    best_class = c;
                }
            }


            if (fabs(((float) (best_class)) - testing_classifications.at<float>( tsample, 0))
                    >= FLT_EPSILON)
            {

                wrong_class++;

                false_positives[best_class]++;

            }
            else
            {
                correct_class++;
            }
        }


        
        for (int i = 0; i < NUMBER_OF_CLASSES; i++)
        {
            printf( "\tClass (digit %d) false postives  %d (%g%%)\n", i,
                    false_positives[i],
                    (double) false_positives[i]*100/NUMBER_OF_TESTING_SAMPLES);
        }
        // all matrix memory free by destructors

        // all OK : main returns 0
*/

}