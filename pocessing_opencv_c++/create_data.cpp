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
#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
//#include <opencv/cxcore.h>
#include "opencv2/core/types_c.h"
//#include <opencv2/core/plot.hpp>


#include "svm.h" // SVM CLASSIFIER
#include "mlp.h" // ARTIFICIAL NEURAL NETWORK
#include "RTree.h"  // RANDOM FOREST CLASSIFIER
#include "BTree.h"  // BOOST 
#include "GBT.h"  // BOOST DESCRIPTOR
#include "multiboost.h"
#include <tr1/memory>
#include "PAuto.h"
#include "featSelectBoost.h"
#include "featSelectGBT.h"
#include "writeMatToFile.h" // SAVE TO FILE
#include "mkKernel.h" // GABOR KERNEL
#include "LBP.h" // LBP DESCRIPTOR RGB 
#include "TDLBP.h"  // LBP DESCRIPTOR DEPTH
#include "LBPD.h"  // LBP DESCRIPTOR DEPTH
//#include "gabormatdepth.h" // GABOR DESCRIPTOR FOR DEPTH
//#include "gabormat.h"  // GABOR DESCRIPTOR RGB 
//#include "hogmat.h" // HOG DESCRIPTOR 
//#include "siftmat.h" // SIFT DESCRIPTOR

using namespace std;
using namespace cv;

//ssh cc-fsr2@10.100.2.234
//qwerty123

int main() {


string part = "face";

//string part = "cheeknose";
//string part = "chin";
//string part = "chin+mouth+jaw";
//string part = "eyes";
//string part = "nose";
//string part = "nosemouth";

//string task = "complexity2";
//string task = "race";
string task = "exp";

string path ="/home/safae/Documents/phd/code/data/Safae/data/"; 
string pathData = path + task + "/" + part + "/";

Mat data,data2,data1lbp,data2lbp, data1sift,data2sift,data1gabor,data2gabor,data1hog,data2hog;
Mat data22lbp,data22sift,data22hog,data22gabor;
Mat image1,im1,im2,im,labels;
Mat imlbp,imhog,imsift,imgabor;
Mat vec1lbp,vec2lbp,vec1sift,vec2sift,vec1gabor,vec2gabor,vec1hog,vec2hog;
Mat labelsN,labelsS,labelsD,labelsSU;
int label,labelN,labelS,labelD,labelSU;
Mat trdata,trlabels;
int n = 3;
//string classes [2] = {"male","female"}; //for gender
//string classes [3] = {"Asian","white","others"}; //for ethnicity
string classes [3] = {"neutral","smile","other"}; // for exp


for(int i=0; i<n; i++){
    
    Mat im1;
    ostringstream stream1,stream2;
    DIR* dir = NULL;
    struct dirent *ent;


    if (classes[i]=="neutral"){
      label = 0;
    }else if(classes[i]=="smile"){
      label = 1;
    }else if(classes[i]=="other"){
      label = 2;
    }

    // model 1
    if (classes[i]=="neutral") labelN = 0;
    else labelN = 1;
    // model 2
    if (classes[i]=="smile") labelS = 0;
    else labelS = 1;
    // model 3
    if (classes[i]=="other") labelD = 0;
    else labelD = 1;

/*
    if (classes[i]=="Asian"){
      label = 0;
    }else if(classes[i]=="white"){
      label = 1;
    }else if(classes[i]=="others"){
      label = 2;
    }
    // model 1
    if (classes[i]=="white") labelN = 0;
    else labelN = 1;
    // model 2
    if (classes[i]=="Asian") labelS = 0;
    else labelS = 1;
    // model 3
    if (classes[i]=="others") labelD = 0;
    else labelD = 1;

    if (classes[i]=="male"){
      label = 0;
    }else if(classes[i]=="female"){
      label = 1;
    }
*/
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
    named = result2.c_str()+name1;  

    //cout << "result1--" << name << endl;
	//cout << "result2--" << named << endl;

    im1 = imread(name, IMREAD_COLOR);
    im2 = imread(named, IMREAD_ANYDEPTH | IMREAD_ANYCOLOR);
    
    
    if(im2.type()!=0){
    cvtColor( im2,im2, COLOR_BGR2GRAY );
    //  equalizeHist( im1,im1 );
       }
    }

    
    //cout <<"DEPTH channels: "<< im2.channels()<<endl;
    //cout <<"RGB channels: "<< im1.channels()<<endl;

    //cout <<"RGB *before* reshape: "<< im1.size()<<endl;
    //cout <<"DEPTH *before* reshape: "<< im2.size()<<endl;


    //im1 = im1.reshape(1,1);
    //im2 = im2.reshape(0,1);

    //cout <<"RGB *after* reshape: "<< im1.size()<<endl;
    //cout <<"DEPTH *after* reshape: "<< im2.size()<<endl;

    //im1.convertTo(im1,CV_32FC1);
    //im2.convertTo(im2,CV_32FC1);

    //normalize(im1,im1, 0, 1, NORM_MINMAX, CV_32F);
    //normalize(im2,im2, 0, 1, NORM_MINMAX, CV_32F); 

    //cout << "normalize success--" << endl;


    //cout <<"RGB image size : "<<im1.size()<<endl;
	//cout <<"DEPTH image size : "<<im2.size()<<endl;
    //hconcat(im1,im2,im);

    //pre-processing

    cvtColor(im1,im1,COLOR_BGR2GRAY);
    equalizeHist(im1,im1);
    medianBlur ( im2, im2, 15 );

    //features extraction

  //LBP(im1,vec1lbp); 
  //vec2lbp = calculate3DLBP(im2);  

    //siftmat(im1,vec1sift);
    //siftmat(im2,vec2sift);

   // gabormat(im1,vec1gabor);
   // gabormatdepth(im2,vec2gabor); 

    //hogmat(im1,vec1hog);
    //hogmat(im2,vec2hog);
   
   // normalize(vec1hog,vec1hog, 0, 1, NORM_MINMAX, CV_32F);
  //normalize(vec2hog,vec2hog, 0, 1, NORM_MINMAX, CV_32F);

    normalize(vec1lbp,vec1lbp, 0, 1, NORM_MINMAX, CV_32F);
    normalize(vec2lbp,vec2lbp, 0, 1, NORM_MINMAX, CV_32F);


   //normalize(vec1sift,vec1sift, 0, 1, NORM_MINMAX, CV_32F);
   //normalize(vec2sift,vec2sift, 0, 1, NORM_MINMAX, CV_32F);

    //normalize(vec1gabor,vec1gabor, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(vec2gabor,vec2gabor, 0, 1, NORM_MINMAX, CV_32F);

   // normalize(vec1hog,vec1hog, 0, 1, NORM_MINMAX, CV_32F);
	//	normalize(vec2hog,vec2hog, 0, 1, NORM_MINMAX, CV_32F);


    
    hconcat(vec1lbp,vec2lbp,imlbp);
   //hconcat(vec1sift,vec2sift,imsift);
   //hconcat(vec1gabor,vec2gabor,imgabor);
	//hconcat(vec1hog,vec2hog,imhog);
    

	normalize(imlbp,imlbp, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(imsift,imsift, 0, 1, NORM_MINMAX, CV_32F);
   //normalize(imhog,imhog, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(imgabor,imgabor, 0, 1, NORM_MINMAX, CV_32F);
	
    //data.push_back(im);
    //data1lbp.push_back(vec1lbp);
    data.push_back(imlbp);


    //data.push_back(imsift);
    //data2sift.push_back(imsift);

    //data1gabor.push_back(vec1gabor);
    //data2gabor.push_back(imgabor);

    //data1hog.push_back(vec1hog);
    //data2hog.push_back(imhog);
 
    labels.push_back(label); // labels matrix   
    labelsN.push_back(labelN);
    labelsS.push_back(labelS);
    labelsD.push_back(labelD);
    //cout <<"LABELS size "<< labels.size()<<endl;

    im1.release(); // FREE MAT
    im2.release();   

    }
  } 
}

	cout << "normalizatino===================>" << endl;
    cout <<"LABELS size "<< labels.size()<<endl;

	//normalize(data,data, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(data1lbp,data1lbp, 0, 1, NORM_MINMAX, CV_32F);
	normalize(data,data, 0, 1, NORM_MINMAX, CV_32F);

	//normalize(data1sift,data1sift, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(data2sift,data2sift, 0, 1, NORM_MINMAX, CV_32F);

	//cout << "type= "<<data2sift.type() << "size = " << data2sift.size() << endl; 
	data = data.reshape(1, data.size().height);
	//cout << "type= "<<data2sift.type() << "size = " << data2sift.size() << endl; 
	
	//normalize(data1hog,data1hog, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(data2hog,data2hog, 0, 1, NORM_MINMAX, CV_32F);

	//normalize(data1gabor,data1gabor, 0, 1, NORM_MINMAX, CV_32F);
	//normalize(data2gabor,data2gabor, 0, 1, NORM_MINMAX, CV_32F);

	/*
	cout << "start FS===================>" << endl;


    Mat id1,id2;
    Mat id1N,id1S,id1D,id1SU,idALL,idALL1,idALL2,idALL3;
    //id1 = featSelectBoost(data,labels);
    id1N = featSelectBoost(data,labelsN);
    id1S = featSelectBoost(data,labelsS);
    id1D = featSelectBoost(data,labelsD);
    //id1SU = featSelectBoost(data,labelsSU);
    
    vconcat(id1N,id1S,idALL1);
    vconcat(idALL1,id1D,idALL3);
    //vconcat(idALL1,id1D,idALL2);
    //vconcat(idALL2,id1SU,idALL3);

    data = data.t(); //transpose
    int s =idALL3.rows;
    cout << "mat/one -----" << idALL3.size() << "===" << s << endl;
    cv::sort(idALL3, idALL3, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);


   std::vector<int> vall;
   std::vector<int> vall1;

    for (int i=0; i<s; ++i){
       
        int a = idALL3.at<int>(i,0);

        if (std::find(vall.begin(),
                      vall.end(),
                      a) == vall.end()) {

            vall.push_back(a);
        }

   }

    idALL = Mat(vall,CV_32S);
    cv::sort(idALL, idALL, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    //cout << "-*-*-FINaL FEATURES*-* "<<  idALL << endl;


    int s1 =idALL.rows;
    cout << "after removing*---" << idALL.size() << "===" << s1 << endl;
     for(int i=0;i<s1;i++) {

         Mat vec;
         int q = idALL.at<int>(i,0);
         
         vec = data.row(q);
         data2.push_back(vec);
      }
 
*/
    data2 = data2.t(); //transpose
/*
cout << "FS2===================>" << endl;
    for(int i=0;i<s2;i++) {
        Mat vec;
        int q = id2.at<int>(i,0);
        vec = data2sift.row(q);
        data22sift.push_back(vec);
    }
    data22sift = data22sift.t();


cout << "FS3===================>" << endl;
    for(int i=0;i<s3;i++) {
        Mat vec;
        int q = id3.at<int>(i,0);
        vec = data2hog.row(q);
        data22hog.push_back(vec);
    }
    data22hog = data22hog.t();


cout << "FS4===================>" << endl;
    for(int i=0;i<s4;i++) {
        Mat vec;
        int q = id4.at<int>(i,0);
        vec = data2gabor.row(q);
        data22gabor.push_back(vec);
    }
    data22gabor = data22gabor.t();
*/

//string filename = path + "labels"+task+".txt";
//writeMatToFile(labels,filename.c_str());

    
cout << "save===================>" << endl;

//string filename1lbp   = path + task + "-" + part + "-lbp-RGBD.txt";
//string filename2lbp   = path + task + "-" + part + "-lbp-DEPTH.txt";

string filename1sift  = path + task + "-" + part + "-sift-RGBD.txt";

//string filename2sift  = path + task + "-" + part + "-sift-DEPTH.txt";

//string filename1gabor = path + task + "-" + part + "-gabor-RGBD.txt";
//string filename2gabor = path + task + "-" + part + "-gabor-DEPTH.txt";

//string filename1hog   = path + task + "-" + part + "-hog-RGBD.txt";

//string filename2hog   = path + task + "-" + part + "-hog-DEPTH.txt";

//writeMatToFile(data2,filename1lbp.c_str()); 
//writeMatToFile(data2lbp,filename2lbp.c_str());

//writeMatToFile(data2,filename1gabor.c_str()); 
//writeMatToFile(data2sift,filename2sift.c_str());

writeMatToFile(data2,filename1sift.c_str()); 
//writeMatToFile(data2hog,filename2hog.c_str());

//writeMatToFile(data22gabor,filename1gabor.c_str()); 
//writeMatToFile(data2gabor,filename2gabor.c_str());


//writeMatToFile(data2gabor2,filename3.c_str());


}
