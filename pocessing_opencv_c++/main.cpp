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
#include <pthread.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

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
#include "modelcreate.h" // SVM CLASSIFIER


#include "svm.h" // SVM CLASSIFIER
#include "mlp.h" // ARTIFICIAL NEURAL NETWORK
#include "RTree.h"  // RANDOM FOREST CLASSIFIER
#include "BTree.h"  // BOOST 
#include "GBT.h"  // BOOST DESCRIPTOR
#include "multiboost.h"
#include <tr1/memory>
#include "PAuto.h"

using namespace std;
using namespace cv;

#define N 1

double max(double a, double b, double c)
{
     double m = a;
     (m < b) && (m = b); //these are not conditional statements.
     (m < c) && (m = c); //these are just boolean expressions.
     return m;
}

struct thread_data {
    
    string path;
    string type;
    PAuto feature;
    string data; 
    Mat telabels;
    Mat predicted;
    float score;

  };
 
void *processthread(void *args)
{
    struct thread_data *data = (struct thread_data *) args;
    cout << "Process data ------"<< data->data << endl;
    modelcreate(data->path,data->type,data->feature,data->data,data->telabels,data->predicted,data->score);
    pthread_exit(NULL);
}

bool XOR(bool a, bool b)
{
    return (a + b) % 2;
}

//int main(int argc, char** argv) {
int main(int argc, char** argv) {


    clock_t tStart = clock();
    string type ="rgb"; //rgb,depth,rgb+depth
    string part[7]={"chin","cheeknose","chin+mouth+jaw","nosemouth","eyes","nose","face"};   

    string path [N] = {part[6]};
    
    string data ="./data/complexity2/"; 
    //string data ="./data/race/";
    // string data ="./data/exp/";
    
   // cout << "------**************-------details ------*********------" <<endl;
    
    string pathdata[N];  
    float score[N];
    Mat telabels[N];
    Mat predicted[N];
    pthread_t thread[N];
    struct thread_data args[N];

    int hs[3]={1363,800,800};
    int MI[3]={9738,10000,9700};
    double lm[3]={0.0927,1, 0.092};
    double sp[3]={0.3,0.4,0.3};
    double bt[3]={1.4586,1,1.4};
    double lr[3]={0.0092,0.001,0.009};


    for (int i = 0; i < N; ++i)
    {

      pathdata[i] = data+path[i]+"/";  //path for data        
      args[i].path = pathdata[i];
      args[i].type = type;

      args[i].feature.Hiddensize=hs[1];
      args[i].feature.lambda=lm[1];
      args[i].feature.sparsityParam=sp[1];
      args[i].feature.beta=bt[1];
      args[i].feature.lrate=lr[1];
      args[i].feature.MI=MI[1];


      //sscanf(argv[1],"hs=%d",&args[i].feature.Hiddensize);
      //sscanf(argv[2],"lm=%lf",&args[i].feature.lambda);
      //sscanf(argv[3],"sp=%lf",&args[i].feature.sparsityParam);
      //sscanf(argv[4],"bt=%lf",&args[i].feature.beta);
      //sscanf(argv[5],"lr=%lf",&args[i].feature.lrate);
      //sscanf(argv[6],"MI=%d",&args[i].feature.MI );
 

      args[i].data=path[i];
      args[i].telabels=telabels[i];
      args[i].predicted=predicted[i];
      args[i].score=score[i];
     
      int rc[N];
      rc[i] = pthread_create(&thread[i], NULL, &processthread, (void *)&args[i]);
     
     }

    pthread_join(thread[0], NULL);
    //pthread_join(thread[1], NULL);
    //pthread_join(thread[2], NULL);
    //pthread_join(thread[3], NULL);
    //pthread_join(thread[4], NULL);
    //pthread_join(thread[5], NULL);
   //pthread_join(thread[6], NULL);     
 
    
//printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

/**********************************PART 2*********************************************/
/*
float score2[N];
Mat telabels2[N];
Mat predicted2[N];
float p[N];
Mat lab;
lab = args[0].telabels;
int s=lab.rows;
Mat result=Mat(s, 1,CV_32F);


for (int j =0; j < s; ++j)

{

double F1=0;
double M1=0;

int n=N-1;
double sum;
Mat tab=Mat(N, 2,CV_32F);
int c;
for (c = 0; c < N ; ++c)
  {
   
   predicted2[c]=args[c].predicted;   // recuperate values
   score2[c]=args[c].score;
   p[c] = predicted2[c].at<float>(j, 0);


    if (p[c]==0){
    M1 = M1+score2[c];
     }else if (p[c]==1){
    F1=F1+score2[c];
 }

}


if (F1<M1){
     result.at<float>(j, 0)=0;
   }else if (F1>M1){
     result.at<float>(j, 0)=1;
   }



}


cout << "------******----------*******-----------GENERAL RESULTS---------*****----------------*****-------"<< endl;

cout <<"testing data size : "<<lab.size()<<endl;

float val;
val=evaluate(result,lab);
cout << "voting decision : "<<val*100<<"%" << endl;

*/
return 0;

}
//end
 
