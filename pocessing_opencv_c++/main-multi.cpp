/**I/O**/

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

/** FUNCTIONS **/

#include "faceDetect.h"// DETECT FACE (STASM LIBRARY)
#include "getLabel.h"  // GET LABELS 
#include "mkKernel.h" // GABOR KERNEL
#include "evaluate.h" // COMPUTE ACCURACY
#include "writeMatToFile.h" // SAVE TO FILE
#include "histImage.h" // COMPUTE HISTOGGRAMME
#include "facedetect.h" // FACE DETECTION (VIOLA & JONES)
#include "getCordinates.h" //PROJECT CORDINATES TO IMAGE

#include "gabormatdepth.h" // GABOR DESCRIPTOR FOR DEPTH
#include "gabormat.h"  // GABOR DESCRIPTOR RGB 
#include "LBP.h" // LBP DESCRIPTOR RGB 
#include "LBPD.h"  // LBP DESCRIPTOR DEPTH
#include "hogmat.h" // HOG DESCRIPTOR 
#include "siftmat.h" // SIFT DESCRIPTOR
#include "cvHaarWavelet.h" // WAVELETTE 
#include "cvInvHaarWavelet.h" // WAVELETTE 
#include "process.h" // WAVELETTE 
#include "featSelectBoost.h"
#include "featSelectGBT.h"
#include "surfmat.h" // SURFT DESCRIPTOR
#include "plot_binary.h" // SVM CLASSIFIER
#include "svm.h" // SVM CLASSIFIER
#include "TDLBP.h" // SVM CLASSIFIER


#include "svm.h" // SVM CLASSIFIER
#include "mlp.h" // ARTIFICIAL NEURAL NETWORK
#include "RTree.h"  // RANDOM FOREST CLASSIFIER
#include "BTree.h"  // BOOST 
#include "GBT.h"  // BOOST DESCRIPTOR
#include "multiboost.h"
#include <tr1/memory>

using namespace std;
using namespace cv;



int main() {

   // parametres setting 

    string classifier="svm";   // classifiers (svm|RF|BF|mlp)
    string I ="rgb+depth";//        // data type (rgb|depth|rgb+depth) 
    string feature ="3DLBP";    // feature extraction method(lbp|sift|gabor|hog|surf|wavelet
    //string pathData = "./data/gender/"; // EUROCOM + Curtin
    string pathData = "./data/race/"; // EUROCOM + Curtin
    string FS = "pca";
    // Initialization // 

    Mat data,image1,im1,image2,im2,labels;
    Mat labelsN,labelsS,labelsD,labelsSU;
    int label,labelN,labelS,labelD,labelSU;
    Mat vec1,vec11,vec01,vect11,vect01,vect1;
    Mat descriptor(14 * 8 * 8 * 4, 1, CV_32F);
    Mat descriptor2(14 * 8 * 8 * 4, 1, CV_32F);
    //hgconst float scaleFactor = 0.05f;


  //string classes [2] = {"male","female"};
     string classes [3] = {"white","Asian","others"};
    //string classes [4] = {"neutral","smile","dis","sur"};


      for(int i=0; i<=2; i++){
          
          ostringstream stream1,stream2;
          DIR* dir = NULL;
          struct dirent *ent;
          
          // get images Path

          if (classes[i]=="white"){
            label = 0;
          }else if(classes[i]=="Asian"){
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
          

/*
   
           if (classes[i]=="neutral"){
            label = 0;
          }else
          
           if(classes[i]=="smile"){
            label = 1;
          }else if(classes[i]=="dis"){
            label = 2;
          }
          else if(classes[i]=="sur"){
            label = 3;
          }
*/
 /*         
          // model 1
          if (classes[i]=="neutral") labelN = 0;
          else labelN = 1;
          // model 2
          if (classes[i]=="smile") labelS = 0;
          else labelS = 1;
          // model 3
          if (classes[i]=="dis") labelD = 0;
          else labelD = 1;
          // model 4
          if (classes[i]=="sur") labelSU = 0;
          else labelSU = 1;
    */      
          stream1 << pathData << classes[i] << "/rgb/" ;
          stream2 << pathData << classes[i] << "/depth/" ;

          string result1 = stream1.str();
          string result2 = stream2.str();

         // cout << "directory ====" << result.c_str() << endl;  
           dir = opendir (result1.c_str());
           if (dir != NULL) {
             while ((ent = readdir (dir)) != NULL) {
                if ( !strcmp(ent->d_name, ".") || !strcmp(ent->d_name, "..") || !strcmp(ent->d_name, "Thumbs.db") ) {
                continue;
                } else {

    
            string name,name1,named;
            name1 = ent->d_name;
            name = result1.c_str() + name1;
            named = result2+name1;
           // cout << "directory actua ====" << name << endl;

            //read images                 
            
            Mat im1,im2,vec1,vec2 ;
            im1 = imread(name,CV_LOAD_IMAGE_COLOR);    
            im2 = imread(named,CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

            //RGB Face normalization : grayscale+intensity normalization for rgb images

            cvtColor(im1,im1,CV_BGR2GRAY);
            equalizeHist(im1,im1);
            //GaussianBlur(im1, im1, Size(7,7), 5, 3, BORDER_CONSTANT);  
            //Depth Face normalization : Smoothing using median filter to eliminate noise

            //imshow("",im1);waitKey();
            //cvtColor( im2,im2, CV_BGR2GRAY );
            //equalizeHist(im2,im2);
            medianBlur ( im2, im2, 15 );  
            //imshow("",im2);waitKey();
            //Feature extraction : Texture - shape meseaures 

            if (I=="rgb") {
                if (feature=="lbp"){
                LBP(im1,vec1);
                }else if (feature=="dwt"){
                // process(im1);
                cvHaarWavelet(im1,vec1,4);
                //imshow("",vec1);waitKey();
                }else if (feature=="sift"){
                siftmat(im1,vec1);
                }else if (feature=="gabor"){
                gabormat(im1,vec1);
                }else if (feature=="hog"){
                hogmat(im1,vec1);
                }else if (feature=="surf"){
                surfmat(im1,vec1);
                }else if (feature=="3DLBP"){
                cv::Mat sel = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(5, 5));
                morphologyEx(im2, im2, MORPH_CLOSE, sel);
                descriptor = calculate3DLBP(im2);  
                vec1=descriptor;      
                }
            }else if (I=="depth") {
                if (feature=="lbp"){
                LBPD(im2,vec1);
                }else if (feature=="sift"){
                siftmat(im2,vec1);
                }else if (feature=="gabor"){
                gabormatdepth(im2,vec1); 
                }else if (feature=="hog"){
                hogmat(im2,vec1);
                }else if (feature=="3DLBP"){
                descriptor = calculate3DLBP(im2);  
                vec1=descriptor;      
                }
          }else if (I=="rgb+depth") {
                if (feature=="lbp"){
             
                LBPD(im2,vec11);LBP(im1,vec01);             
                }else if (feature=="sift"){
                siftmat(im2,vec11);siftmat(im1,vec01);
                }else if (feature=="gabor"){
                gabormatdepth(im2,vec11);gabormat(im1,vec01);
                vec11 = vec11.reshape(1, 1);vec11.convertTo(vec11,CV_32FC1); 
                }else if (feature=="hog"){
                hogmat(im2,vec11);hogmat(im1,vec01); 
                }else if (feature=="3DLBP"){
                descriptor = calculate3DLBP(im1);  
                descriptor2 = calculate3DLBP(im2);  
                vec11=descriptor; 
                vec01=descriptor2; 
                }
                else if (feature=="hog+lbp"){
                hogmat(im2,vec11);gabormat(im1,vec01);
                //vec01.convertTo(vec01,CV_32FC1);  
                //cout <<"depth type ===== " << vec11.type()<< endl;
                //cout <<"rgb type ===== "<< vec01.type()<< endl;
                }   

            hconcat(vec01,vec11,vec1);

            }
            
            im1.release(); // FREE MAT
            im2.release();      

        data.push_back(vec1); // training data matrix
        labels.push_back(label); // labels matrix
        labelsN.push_back(labelN);
        labelsS.push_back(labelS);
        labelsD.push_back(labelD);
        labelsSU.push_back(labelSU);  
       // j++;
      }
    }
  }  
}   

  //cout << "labels======>" << labels<< endl;
    cout << "data type =====" << data.type() << endl;
    data = data.reshape(1, data.size().height);
    data.convertTo(data,CV_32FC1);
    cout << "data size ====="<< feature << "=="<<  I  << "======="<< data.size() << endl; 
    //cout << "tr ======" << trdata.size() << "test=======" << testdata.size() << endl;
    //cout << "tr ======" << labels.size() << "test=======" << telabels.size() << endl;

/**************FEATURE SELECTION***********************/
   cout << "type of data----" << data.type() << endl;
   data.convertTo(data,CV_32FC1, 1.0/255, 0);  
   normalize(data,data, 0, 1, NORM_MINMAX, CV_32F);
   cout << "type of data*****" << data.type() << endl;
   Mat data2;

 
  if (FS=="pca") {
      PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW,.95); // variance retrained value    
      pca.project(data,data2);
      data2.convertTo(data2,CV_32FC1, 1.0/255, 0);
      normalize(data2,data2, 0, 1, NORM_MINMAX, CV_32F);
      cout << "---FEATURES after selection PCA ====="<< data2.size() << endl; 
      //cout << "---FEATURES after selection TESTING ====="<< testdata2.size() << endl;              
  }else if (FS=="boost"){
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
           /* code */
      }
 

    data2 = data2.t(); //transpose
        
    //cout << "-----" << data2.size() << endl;
 }else if (FS=="GBT"){
    Mat id1,id2;
    id1 = featSelectGBT(data,labels);
    data = data.t(); //transpose
    int s =id1.rows;
   // cout << "" << id1.size() << "===" << s << endl; 
     for(int i=0;i<s;i++) {
         Mat vec;
         int q = id1.at<int>(i,0);
         //cout << "col element ==" << q << endl;
         vec = data.row(q);
         data2.push_back(vec);
        // cout << "-----" << data2.size() << endl;
      }

    data2 = data2.t(); //transpose
    //cout << "-----" << data2.size() << endl;
 }else{

     data2=data; //no feature selection
   }
   //cout << "labels --"<< labels<< endl;
 //writeMatToFile(data2,);
 // cout << "STEP Three ======>go go go" << endl;
/*****************************data ===******************************/
  cv::Mat trdata,testdata;
  cv::Mat trlabels,telabels;

  /*
   for(int l=0;l<313;l++) {
            Mat vec,lab;
            vec = data2.row(l);
            lab = labels.row(l);
            trdata.push_back(vec); 
            trlabels.push_back(lab); 
           // cout << "l==" << l << "===" <<  lab<< endl;

    }

   for(int k=313;k<489;k++) {
            Mat vecc,labb;
            vecc = data2.row(k);
            labb = labels.row(k);
            testdata.push_back(vecc); 
            telabels.push_back(labb); 
     // cout << "FEMALE"  << c << "==="<< labb<< endl; 
    }   
  
  for(int c=489;c<572;c++) {
            Mat vecc,labb;
            vecc = data2.row(c);
            labb = labels.row(c);
            trdata.push_back(vecc); 
            trlabels.push_back(labb); 
     // cout << "FEMALE"  << c << "==="<< labb<< endl; 
    }

*/
    
   for(int l=0;l<146;l++) {
            Mat vec,lab;
            vec = data2.row(l);
            lab = labels.row(l);
            trdata.push_back(vec); 
            trlabels.push_back(lab); 
           // cout << "l==" << l << "===" <<  lab<< endl;

    }

   for(int k=146;k<274;k++) {
            Mat vecc,labb;
            vecc = data2.row(k);
            labb = labels.row(k);
            testdata.push_back(vecc); 
            telabels.push_back(labb); 
     //cout << "FEMALE"  << c << "==="<< labb<< endl; 
    }   


  for(int c=274;c<538;c++) {
            Mat vecc,labb;
            vecc = data2.row(c);
            labb = labels.row(c);
            trdata.push_back(vecc); 
            trlabels.push_back(labb); 
     //cout << "FEMALE"  << c << "==="<< labb<< endl; 
    }

     for(int k=538;k<583;k++) {
            Mat vecc,labb;
            vecc = data2.row(k);
            labb = labels.row(k);
            testdata.push_back(vecc); 
            telabels.push_back(labb); 
     // cout << "FEMALE"  << c << "==="<< labb<< endl; 
  
    } 

    /*
  for(int l=0;l<73;l++) {
            Mat vec,lab;
            vec = data2.row(l);
            lab = labels.row(l);
            trdata.push_back(vec); 
            trlabels.push_back(lab); 
           // cout << "l==" << l << "===" <<  lab<< endl;

    }

   for(int k=73;k<135;k++) {
            Mat vecc,labb;
            vecc = data2.row(k);
            labb = labels.row(k);
            testdata.push_back(vecc); 
            telabels.push_back(labb); 
     //cout << "FEMALE"  << c << "==="<< labb<< endl; 
    }   


  for(int c=135;c<281;c++) {
            Mat vecc,labb;
            vecc = data2.row(c);
            labb = labels.row(c);
            trdata.push_back(vecc); 
            trlabels.push_back(labb); 
     //cout << "FEMALE"  << c << "==="<< labb<< endl; 
    }

     for(int k=281;k<343;k++) {
            Mat vecc,labb;
            vecc = data2.row(k);
            labb = labels.row(k);
            testdata.push_back(vecc); 
            telabels.push_back(labb); 
     // cout << "FEMALE"  << c << "==="<< labb<< endl; 
}
     for(int k=343;k<416;k++) {
            Mat vecc,labb;
            vecc = data2.row(k);
            labb = labels.row(k);
            trdata.push_back(vecc); 
            trlabels.push_back(labb); 
     // cout << "FEMALE"  << c << "==="<< labb<< endl; 
}

*/

    cout << "=====rtrlabels "<< trlabels.size() <<  endl;
    cout << "=====testlabels "<< telabels.size() <<  endl;
    cout << "=====TRAINING "<< trdata.size() <<  endl;
    cout << "=====TESTING"<< testdata.size() <<  endl;

  if (classifier=="svm"){
    cout << "---GO GO SVM" << endl;
    svm(trdata,trlabels,testdata,telabels);
    cout << "---GO GO RF" << endl;
    RTree(trdata,trlabels,testdata,telabels);
    cout << "---GO GO GBT" << endl;
    GBT(trdata,trlabels,testdata,telabels);

  }else if (classifier=="RF"){
      RTree(trdata,trlabels,testdata,telabels);
  }
  else if (classifier=="BF"){    
      BTree(trdata,trlabels,testdata,telabels);
  }else if (classifier=="GBT"){    
     GBT(trdata,trlabels,testdata,telabels);
  }else if (classifier=="bayes"){
    
    CvNormalBayesClassifier *bayes = new CvNormalBayesClassifier;
    bayes->train(trdata, trlabels, Mat(), Mat(), false);
    cv::Mat predicted(telabels.rows, 1, CV_32F);
    for (int i = 0; i < testdata.rows; i++) {
    const cv::Mat sample = testdata.row(i);
    predicted.at<float>(i, 0) = bayes->predict(sample);
    cout <<">>> / "<<telabels.row(i) <<" // "<< bayes->predict(sample)<<endl;
     }
  }
  else if (classifier=="MB"){
  multiboost(trdata,trlabels,testdata,telabels);
}


return 0;

}
//end
 