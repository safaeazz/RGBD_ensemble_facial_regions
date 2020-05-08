#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include "opencv2/gpu/gpu.hpp"

#include <math.h>
#include <cmath>
#include <cstdio>

using namespace cv;
using namespace std;

//cv::Mat faceDetect(cv::Mat& image,float landmarks[2*77])
void faceDetect(cv::Mat& image,cv::Mat& face,float landmarks[2*77])

{
  double wf =abs(landmarks[11*2]-landmarks[1*2]);
  double hf =abs(landmarks[14*2+1]-landmarks[6*2+1]); 
  cv::Point ef(landmarks[1*2], landmarks[14*2+1]);

  face = image(Rect(ef.x,ef.y,wf,hf));
 // imshow("",face);waitKey();
/*
  double w =abs(landmarks[11*2]-landmarks[1*2]);
  double h =abs(landmarks[14*2+1]-landmarks[6*2+1]); 
  cv::Point e(landmarks[1*2], landmarks[14*2+1]);    
  image = image(Rect(e.x,e.y,w,h));
  return image;
*/
}