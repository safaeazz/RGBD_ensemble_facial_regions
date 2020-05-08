/**I/O

TOdo
irace - sparse autoencoders
-denoising autoencoders
-SDA
-CAE/CNN
see the tabs in chrome
PDF desktop

**/

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

#include "svm.h" // SVM CLASSIFIER
#include "mlp.h" // ARTIFICIAL NEURAL NETWORK
#include "RTree.h"  // RANDOM FOREST CLASSIFIER
#include "BTree.h"  // BOOST 
#include "GBT.h"  // BOOST DESCRIPTOR
#include "multiboost.h"
#include <tr1/memory>
#include "PAuto.h"

/** FUNCTIONS **/

using namespace cv;
using namespace std;

#define IS_TEST 0
#define IS_TEST_SA 0
#define IS_TEST_SMR 0
#define IS_TEST_FT 0

#define ATD at<double>
#define elif else if

int SparseAutoencoderLayers = 2;
int nclasses = 2;
int batch;


typedef struct SparseAutoencoder{
    Mat W1;
    Mat W2;
    Mat b1;
    Mat b2;
    Mat W1grad;
    Mat W2grad;
    Mat b1grad;
    Mat b2grad;
    double cost;
}SA;

typedef struct SparseAutoencoderActivation{
    Mat aInput;
    Mat aHidden;
    Mat aOutput;
}SAA;



Mat 
sigmoid(Mat &M){
    Mat temp;
    exp(-M, temp);
    return 1.0 / (temp + 1.0);
}

Mat 
dsigmoid(Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

void
weightRandomInit(SA &sa, int inputsize, int hiddensize, int nsamples, double epsilon){

    double *pData;
    sa.W1 = Mat::ones(hiddensize, inputsize, CV_64FC1);
    for(int i=0; i<hiddensize; i++){
        pData = sa.W1.ptr<double>(i);
        for(int j=0; j<inputsize; j++){
            pData[j] = randu<double>();
        }
    }
    sa.W1 = sa.W1 * (2 * epsilon) - epsilon;

    sa.W2 = Mat::ones(inputsize, hiddensize, CV_64FC1);
    for(int i=0; i<inputsize; i++){
        pData = sa.W2.ptr<double>(i);
        for(int j=0; j<hiddensize; j++){
            pData[j] = randu<double>();
        }
    }
    sa.W2 = sa.W2 * (2 * epsilon) - epsilon;


    sa.b1 = Mat::ones(hiddensize, 1, CV_64FC1);
    for(int j=0; j<hiddensize; j++){
        sa.b1.ATD(j, 0) = randu<double>();
    }
    sa.b1 = sa.b1 * (2 * epsilon) - epsilon;

    sa.b2 = Mat::ones(inputsize, 1, CV_64FC1);
    for(int j=0; j<inputsize; j++){
        sa.b2.ATD(j, 0) = randu<double>();
    }
    sa.b2 = sa.b2 * (2 * epsilon) - epsilon;



    sa.W1grad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    sa.W2grad = Mat::zeros(inputsize, hiddensize, CV_64FC1);
    sa.b1grad = Mat::zeros(hiddensize, 1, CV_64FC1);
    sa.b2grad = Mat::zeros(inputsize, 1, CV_64FC1);
    sa.cost = 0.0;
}


SAA
getSparseAutoencoderActivation(SA &sa, Mat &data){
    SAA acti;
    data.copyTo(acti.aInput);
    acti.aHidden = sa.W1 * acti.aInput + repeat(sa.b1, 1, data.cols);
    acti.aHidden = sigmoid(acti.aHidden);
    acti.aOutput = sa.W2 * acti.aHidden + repeat(sa.b2, 1, data.cols);
    acti.aOutput = sigmoid(acti.aOutput);
    return acti;
}

void
sparseAutoencoderCost(SA &sa, Mat &data, double lambda, double sparsityParam, double beta){

    int nfeatures = data.rows;
    int nsamples = data.cols;
    SAA acti = getSparseAutoencoderActivation(sa, data);

    Mat errtp = acti.aOutput - data;
    pow(errtp, 2.0, errtp);
    errtp /= 2.0;
    double err = sum(errtp)[0] / nsamples;
    // now calculate pj which is the average activation of hidden units
    Mat pj;
    reduce(acti.aHidden, pj, 1, CV_REDUCE_SUM);
    pj /= nsamples;
    // the second part is weight decay part
    double err2 = sum(sa.W1)[0] + sum(sa.W2)[0];
    err2 *= (lambda / 2.0);
    // the third part of overall cost function is the sparsity part
    Mat err3;
    Mat temp;
    temp = sparsityParam / pj;
    log(temp, temp);
    temp *= sparsityParam;
    temp.copyTo(err3);
    temp = (1 - sparsityParam) / (1 - pj);
    log(temp, temp);
    temp *= (1 - sparsityParam);
    err3 += temp;
    sa.cost = err + err2 + sum(err3)[0] * beta;

    // following are for calculating the grad of weights.
    Mat delta3 = -(data - acti.aOutput);
    delta3 = delta3.mul(dsigmoid(acti.aOutput));
    Mat temp2 = -sparsityParam / pj + (1 - sparsityParam) / (1 - pj);
    temp2 *= beta;
    Mat delta2 = sa.W2.t() * delta3 + repeat(temp2, 1, nsamples);
    delta2 = delta2.mul(dsigmoid(acti.aHidden));
    Mat nablaW1 = delta2 * acti.aInput.t();
    Mat nablaW2 = delta3 * acti.aHidden.t();
    Mat nablab1, nablab2; 
    delta3.copyTo(nablab2);
    delta2.copyTo(nablab1);
    sa.W1grad = nablaW1 / nsamples + lambda * sa.W1;
    sa.W2grad = nablaW2 / nsamples + lambda * sa.W2;
    reduce(nablab1, sa.b1grad, 1, CV_REDUCE_SUM);
    reduce(nablab2, sa.b2grad, 1, CV_REDUCE_SUM);
    sa.b1grad /= nsamples;
    sa.b2grad /= nsamples;
}

void
gradientChecking(SA &sa, Mat &data, double lambda, double sparsityParam, double beta){

    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    sparseAutoencoderCost(sa, data, lambda, sparsityParam, beta);
    Mat w1g(sa.W1grad);
   // cout<<"test sparse autoencoder !!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<sa.W1.rows; i++){
        for(int j=0; j<sa.W1.cols; j++){
            double memo = sa.W1.ATD(i, j);
            sa.W1.ATD(i, j) = memo + epsilon;
            sparseAutoencoderCost(sa, data, lambda, sparsityParam, beta);
            double value1 = sa.cost;
            sa.W1.ATD(i, j) = memo - epsilon;
            sparseAutoencoderCost(sa, data, lambda, sparsityParam, beta);
            double value2 = sa.cost;
            double tp = (value1 - value2) / (2 * epsilon);
          //  cout<<i<<", "<<j<<", "<<tp<<", "<<w1g.ATD(i, j)<<", "<<w1g.ATD(i, j) / tp<<endl;
            sa.W1.ATD(i, j) = memo;
        }
    }
}

void
trainSparseAutoencoder(SA &sa, Mat &data, int hiddenSize, double lambda, double sparsityParam, double beta, double lrate, int MaxIter){

    int nfeatures = data.rows;
    int nsamples = data.cols;
    weightRandomInit(sa, nfeatures, hiddenSize, nsamples, 0.12);
    if (IS_TEST_SA){
        gradientChecking(sa, data, lambda, sparsityParam, beta);
    }else{
        int converge = 0;
        double lastcost = 0.0;
       // cout<<"Sparse Autoencoder Learning: "<<endl;
        while(converge < MaxIter){

            int randomNum = rand() % (data.cols - batch);
            Rect roi = Rect(randomNum, 0, batch, data.rows);
            Mat batchX = data(roi);

            sparseAutoencoderCost(sa, batchX, lambda, sparsityParam, beta);
         //  cout<<"learning step: "<<converge<<", Cost function value = "<<sa.cost<<", randomNum = "<<randomNum<<endl;
            if(fabs((sa.cost - lastcost) ) <= 5e-5 && converge > 0) break;
            if(sa.cost <= 0.0) break;
            lastcost = sa.cost;
            sa.W1 -= lrate * sa.W1grad;
            sa.W2 -= lrate * sa.W2grad;
            sa.b1 -= lrate * sa.b1grad;
            sa.b2 -= lrate * sa.b2grad;
            ++ converge;
        }
    }
}



void modelcreate(string pathData, string type,PAuto feature,string nameclassifier,cv::Mat& telabels,Mat& predicted,float& score){
    
    Mat data,image1,im1,im2,im,labels;
    int label;
    Mat trdata,trlabels;
    string classes [2] = {"male","female"}; //for gender
           
     for(int i=0; i<2; i++){

          Mat im1;
          ostringstream stream1,stream2;
          DIR* dir = NULL;
          struct dirent *ent;

          if (classes[i]=="male"){
            label = 0;
          }else if(classes[i]=="female"){
            label = 1;
          }
    
          stream1 << pathData << classes[i] << "/rgb/" ;
          stream2 << pathData << classes[i] << "/depth/" ;


          string result1 = stream1.str();
          string result2 = stream2.str();
          
           dir = opendir (result1.c_str());
           if (dir != NULL) {
             while ((ent = readdir (dir)) != NULL) {
                if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..") || !strcmp(ent->d_name, "Thumbs.db") ) {
                continue;
                } else {

            string name,name1,named;
            name1 = ent->d_name;
            name = result1.c_str() + name1;
            named = result2.c_str()+ name1;  

            im1 = imread(name, CV_LOAD_IMAGE_COLOR);
            im2 = imread(named, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
            
             if(im2.type()!=0){
             cvtColor( im2,im2, CV_BGR2GRAY );
           //  equalizeHist( im1,im1 );
             }
          }
          
          im1 = im1.reshape(1,1);
          im2 = im2.reshape(0,1);
          im1.convertTo(im1,CV_32FC1);
          im2.convertTo(im2,CV_32FC1);

          normalize(im1,im1, 0, 1, NORM_MINMAX, CV_32F);
          normalize(im2,im2, 0, 1, NORM_MINMAX, CV_32F); 
          hconcat(im1,im2,im);
         // normalize(im,im, 0, 1, NORM_MINMAX, CV_32F); 
//
          data.push_back(im2);
          labels.push_back(label); // labels matrix   

          \\


      } 
    }
  }
        

        Mat testdata;
        data = data.t(); //transpose because opencv take
        data.convertTo(data,CV_64FC2);

        Mat trainX, trainY;
        trainX=data;
        normalize(trainX,trainX, 0, 1, NORM_MINMAX, CV_64FC2);
        Mat testX, testY;
        batch = trainX.cols / 100;
       // cout << "batch--" << batch << endl;
        
       // Finished reading data  
       // pre-processing data. 
                
       //Scalar mean, stddev;
       //meanStdDev(trainX, mean, stddev);
       //Mat normX = trainX - mean[0]; 
       //normX.copyTo(trainX);

       //cout << "-*-*-*-*--*-*********************************************" << endl;
        vector<SA> HiddenLayers;
        vector<Mat> Activations;
        for(int i=0; i<SparseAutoencoderLayers; i++){
          Mat tempX;
          if(i == 0) trainX.copyTo(tempX); else Activations[Activations.size() - 1].copyTo(tempX);
          SA tmpsa;;
          trainSparseAutoencoder(tmpsa, tempX, feature.Hiddensize, feature.lambda, feature.sparsityParam, feature.beta, feature.lrate,feature.MI);
//trainSparseAutoencoder(tmpsa, tempX, 1000, 3e-3, 0.1, 3, 2e-2, 100);     
//(&sa,&data,hiddenSize, lambda, sparsityParam,  beta, double lrate, int MaxIter){
//L2 Weight Regularization ( Lambda) Sparsity Regularization (Beta) Sparsity proportion (Rho).
          Mat tmpacti = tmpsa.W1 * tempX + repeat(tmpsa.b1, 1, tempX.cols);
          tmpacti = sigmoid(tmpacti);
          HiddenLayers.push_back(tmpsa);
          Activations.push_back(tmpacti);
          data=Activations[Activations.size() - 1];
        }
        //cout << "------------" << endl;
        data = data.t();
        data.convertTo(data,CV_32FC1);

        Mat test,test2;
        for(int l=0;l<200;l++) {
            Mat vec,lab;
            vec = data.row(l);
            lab = labels.row(l);
            trdata.push_back(vec); 
            trlabels.push_back(lab); 

          }

        for(int k=200;k<406;k++) {
            Mat vecc,labb;
            vecc = data.row(k);
            labb = labels.row(k);
            test.push_back(vecc); 
            test2.push_back(labb); 

          }   
  
        for(int c=406;c<468;c++) {
            Mat vecc,labb;
            vecc = data.row(c);
            labb = labels.row(c);
            trdata.push_back(vecc); 
            trlabels.push_back(labb); 
         }

        //cout << "------tracout << "batch--" << batch << endl;ining------" << endl;

        testdata=test;  
        telabels=test2;
        test.release();
        test2.release();  

      svm(trdata,trlabels,nameclassifier,testdata,telabels,predicted,score);

    // Finished training Sparse Autoencoder

}   
    

