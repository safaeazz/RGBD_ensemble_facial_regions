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
//#include <opencv2/nonfree/features2d.hpp> 
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "histImage.h"
#include "hogmat.h"

using namespace cv;
using namespace std;

void hogmat(cv::Mat& image, cv::Mat& vec1) {

/***********************************************************CPU_VERSION*************************************/
	
	vector<float> hog_features;
	if (image.type()!= CV_8UC4 || image.type()!= CV_8UC3){
		
		image.convertTo(image, CV_8U);
	}
  	

  	//HOGDescriptor hog(Size(16, 16), Size(16, 16), Size(16, 16), Size(16, 16),8, 0,-1, 0, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
	//cvtColor(image, image, CV_RGB2GRAY);
	// equalizeHist(image,image); 
	
	HOGDescriptor hog = HOGDescriptor(cvSize(16,16),cvSize(16,16), cvSize(16,16),cvSize(16,16),8);
	hog.compute(image, hog_features,Size(10,10), Size(40,40));
  	//trainingdata.push_back(hog_features);
	//cout << hog_features.size() << endl;
	
	Mat vectx = Mat(hog_features,CV_8UC4);
	vec1 = vectx.reshape(0,1);
  //cout << vec1.size() << endl;

/***********************************GPU_VERSION*************************************/

/*
cv::Mat temp;
gpu::GpuMat gpu_img, descriptors;
cv::gpu::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                               cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                               cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
gpu_img.upload(img);
gpu_hog.getDescriptors(gpu_img, win_stride, descriptors, cv::gpu::HOGDescriptor::DESCR_FORMAT_ROW_BY_ROW);
descriptors.download(temp);

CV_8U	CV_8S CV_16U	CV_16S	CV_32S	CV_32F	CV_64F
*/

}
