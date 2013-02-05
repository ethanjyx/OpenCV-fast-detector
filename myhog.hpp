#ifndef __OPENCV_MYHOG_HPP__
#define __OPENCV_MYHOG_HPP__

#include "opencv2/objdetect/objdetect.hpp"

namespace cv
{

struct CV_EXPORTS_W myHOGDescriptor : HOGDescriptor{	

	vector<Mat> fftSvmDetector;

	CV_WRAP virtual void compute(const Mat& img,
		CV_OUT vector<float>& descriptors,
		Size winStride=Size(), Size padding=Size(),
		const vector<Point>& locations=vector<Point>()) const;				

	CV_WRAP virtual void detect(const Mat& img, 
		CV_OUT vector<Point>& foundLocations, 
		CV_OUT vector<double>& weights,
		double hitThreshold=0, Size winStride=Size(),
		Size padding=Size(),
		const vector<Point>& searchLocations=vector<Point>()) const;
    
	virtual void detect(const Mat& img, CV_OUT vector<Point>& foundLocations,
                        double hitThreshold=0, Size winStride=Size(),
                        Size padding=Size(),
                        const vector<Point>& searchLocations=vector<Point>()) const;
	//with result weights output

    CV_WRAP void detectMultiScale(
		const Mat& img, CV_OUT vector<Rect>& foundLocations,
		CV_OUT vector<double>& foundWeights, double hitThreshold=0,
		Size winStride=Size(), Size padding=Size(), double scale=1.05,
		double finalThreshold=2.0,bool useMeanshiftGrouping = false) const;
	
	//without found weights output
	void detectMultiScale(
		const Mat& img, CV_OUT vector<Rect>& foundLocations,
		double hitThreshold=0, Size winStride=Size(),
        Size padding=Size(), double scale=1.05,
		double finalThreshold=2.0, bool useMeanshiftGrouping = false) const;
	void placeDetector();
};

}

#endif
