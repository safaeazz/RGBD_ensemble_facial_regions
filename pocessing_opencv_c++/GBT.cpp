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
#include "GBT.h"


using namespace std;
using namespace cv;

void GBT(cv::Mat& trainingdata, cv::Mat& trainingClasses,string name, cv::Mat& testdata, cv::Mat& testClasses,cv::Mat& predicted,float& score)
//void GBT(cv::Mat& trainingdata, cv::Mat& trainingClasses,cv::Mat& testdata, cv::Mat& testClasses)
{
  
 cv::Mat var_type = cv::Mat(trainingdata.size().width +1, 1, CV_8UC1);
  var_type.setTo(Scalar(CV_VAR_NUMERICAL) ); // all inputs are numerical
  var_type.at<uchar>(trainingdata.size().width, 0) = CV_VAR_CATEGORICAL; //CV_VAR_ORDERED
  //float priors[] = {1,1,1};  // weights of each classification for classes


 CvGBTrees gbt;
CvGBTreesParams params = CvGBTreesParams(CvGBTrees::DEVIANCE_LOSS, 3000, 0.95f, 0.001f, 25, false);
    //: CvDTreeParams( 3, 10, 0, false, 10, 0, false, false, 0 )
 gbt.train( trainingdata, CV_ROW_SAMPLE, trainingClasses,  Mat(), Mat(), var_type,
                       Mat(), params, false);
  //boost->train( trainingdata, CV_ROW_SAMPLE, trainingClasses,  Mat(), Mat(), var_type,
  //boost->save("./files/rtree.xml");
  //boost->load("./files/rtree.xml");

  predicted= Mat(testClasses.rows, 1,CV_32F);

  for (int i = 0; i < testdata.rows; i++) {
    const cv::Mat sample = testdata.row(i);

    predicted.at<float>(i, 0) = gbt.predict(sample);
  }
  score=evaluate(predicted,testClasses);
  cout << "---GBT" << score << endl;


}
    