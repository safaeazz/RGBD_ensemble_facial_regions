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
#include "plot_binary.h"

using namespace std;
using namespace cv;
//, cv::Mat& testData, cv::Mat& testClasses

//void svm(cv::Mat& trainingData, cv::Mat& trainingClasses,string name) {
//void svm(cv::Mat& trainingData, cv::Mat& trainingClasses,string name, cv::Mat& testData, cv::Mat& testClasses,float& score){
void svm(cv::Mat& trainingData, cv::Mat& trainingClasses,string name, cv::Mat& testData, cv::Mat& testClasses,cv::Mat& predicted,float& score){
	
	CvSVMParams param = CvSVMParams();
	param.svm_type = CvSVM::C_SVC;
	param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
	param.degree = 0; // for poly
	param.gamma = 0.1; // for poly/rbf/sigmoid
	param.coef0 = 0; // for poly/sigmoid

	param.C = 10; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
	param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR

	param.class_weights = NULL; // for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER +CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;

	CvSVM svm;
	//svm.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), param);

svm.train_auto(trainingData, trainingClasses, Mat(), Mat(), param,10);
//CvSVMParams p;
//p= svm.get_params() ;
   //printf( "\nUsing optimal parameters degree %f, gamma %f, ceof0 %f\n\t C %f, nu %f, p %f\n Training ..",
              //  p.degree, p.gamma, p.coef0, p.C, p.nu, p.p);
//cout << "    "<< endl;
//string dest;
//dest = "./files/"+name+".xml";
//svm.save(dest.c_str());

Mat responses;
svm.predict( testData, responses );

	int t = 0;
	int f = 0;
	
	predicted= Mat(testClasses.rows, 1,CV_32F);
	for(int i = 0; i < testData.rows; i++) {
         
		cv::Mat sample = testData.row(i);
		predicted.at<float>(i, 0) = svm.predict(sample);
   // cout <<">>> / "<<testClasses.row(i) <<" // "<<svm.predict(sample)<<endl;

           
	}
score=evaluate(predicted,testClasses);
 //score=-1*score;
cout << "" << score << endl;

 
}
