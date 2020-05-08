#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


#include "computeVF.h"

using namespace cv;
using namespace std;


int * computeVF(string path)
{

   
    Mat image;
    image = imread(path, CV_LOAD_IMAGE_COLOR);   // Read the file
    //cout << "M = "<< endl << " "  << image <<  endl << endl ;

    
 

    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", image );                   // Show our image inside it.

    //waitKey(0);  
    

                                            // Wait for a keystroke in the window
   return NULL;
}