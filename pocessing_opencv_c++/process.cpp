#include <iostream>
#include <math.h>
#include <cmath>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv/cv.h>
#include "cvHaarWavelet.h"
#include "cvInvHaarWavelet.h"

using namespace std;
using namespace cv;

#define NONE 0  // no filter
#define HARD 1  // hard shrinkage
#define SOFT 2  // soft shrinkage
#define GARROT 3  // garrot filterr
//--------------------------------

int process(Mat& frame)
{
    int n = 0;
    const int NIter=4;
    char filename[200];

    Mat GrayFrame;
    Mat Src;
    Mat Dst;
    Mat Temp;
    Mat Filtered;
    for (;;) 
    {
     
        if (frame.empty()) continue;
        cvtColor(frame, GrayFrame, CV_BGR2GRAY);
        cout <<"step.1==" << endl;
        imshow("one", GrayFrame);waitKey();
        cout << GrayFrame.type()<< endl;
        GrayFrame.convertTo(GrayFrame,CV_32FC1);
        imshow("one", GrayFrame);waitKey();
        cvHaarWavelet(GrayFrame,Dst,NIter);
        cout <<"step1==" << endl;

        Dst.copyTo(Temp);

        cvInvHaarWavelet(Temp,Filtered,NIter,GARROT,30);

        imshow("one", frame);waitKey();

        double M=0,m=0;
        //----------------------------------------------------
        // Normalization to 0-1 range (for visualization)
        //----------------------------------------------------
        minMaxLoc(Dst,&m,&M);
        if((M-m)>0) {Dst=Dst*(1.0/(M-m))-m/(M-m);}
        imshow("Coeff", Dst);

        minMaxLoc(Filtered,&m,&M);
        if((M-m)>0) {Filtered=Filtered*(1.0/(M-m))-m/(M-m);}        
        imshow("Filtered", Filtered);
/*
        char key = (char)waitKey(5);
        switch (key) 
        {
        case 'q':
        case 'Q':
        case 27: //escape key
            return 0;
        case ' ': //Save an image
            sprintf(filename,"filename%.3d.jpg",n++);
            imwrite(filename,frame);
            cout << "Saved " << filename << endl;
            break;
        default:
            break;
        }
*/
    }
    return 0;
}