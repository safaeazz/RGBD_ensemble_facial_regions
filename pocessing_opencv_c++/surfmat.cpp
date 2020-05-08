#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>

#include "surfmat.h"

using namespace cv;
using namespace std;

void surfmat(cv::Mat& image, Mat& vec1)
{

//cvtColor(image,image, CV_BGR2GRAY );

	vector<KeyPoint> keypoints;

	Mat output;
	SurfFeatureDetector surf(2500);
	surf.detect(image, keypoints);
	drawKeypoints(image, keypoints, output, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//namedWindow("SURF detector img1");
	//imshow("SURF detector img1", output);


	SurfDescriptorExtractor surfDesc;
	Mat descriptors;
	surfDesc.compute(image, keypoints, descriptors);
	cout << descriptors.size() << endl;

	//vec1=descriptors;

	vec1 = output.reshape(0,1);
}





