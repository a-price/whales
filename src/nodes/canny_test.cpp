/**
 * \file random_spawn.cpp
 * \brief Creates a random pile of stuff to play with
 *
 * \author Andrew Price
 * \date September 13, 2013
 *
 * \copyright
 *
 * Copyright (c) 2013, Georgia Tech Research Corporation
 * All rights reserved.
 *
 * Humanoid Robotics Lab Georgia Institute of Technology
 * Director: Mike Stilman http://www.golems.org
 *
 * This file is provided under the following "BSD-style" License:
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include <string>
#include <fstream>
#include <streambuf>

#include <ros/ros.h>
#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "fh_segmentation/fh_segmentation.h"


const cv::Scalar keypointColor = cv::Scalar(0, 255, 0); // Green Keypoints

std::vector<std::string> objectNames;

ros::Publisher zPub;

ros::ServiceClient worldInfoClient;
ros::ServiceClient spawnClient;
ros::ServiceClient deleteClient;

float randbetween(float min, float max)
{
	return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

class ColorGenerator
{
public:
	static double interpolate( double val, double y0, double x0, double y1, double x1 )
	{
		return (val-x0)*(y1-y0)/(x1-x0) + y0;
	}

	static double base( double val )
	{
		if ( val <= -0.75 ) return 0;
		else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
		else if ( val <= 0.25 ) return 1.0;
		else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
		else return 0.0;
	}

	static double red( double gray )	{ return base( gray - 0.5 );	}
	static double green( double gray )	{ return base( gray );	}
	static double blue( double gray )	{ return base( gray + 0.5 );	}

	static cv::Scalar jet(float val, float minVal = -1, float maxVal = 1)
	{
		float scaledVal = ((maxVal-minVal) * val) + minVal;
		return cv::Scalar(blue(scaledVal)*255, green(scaledVal)*255, red(scaledVal)*255);
	}

};

static const cv::Vec3b bcolors[] =
{
	cv::Vec3b(0,0,255),
	cv::Vec3b(0,128,255),
	cv::Vec3b(0,255,255),
	cv::Vec3b(0,255,0),
	cv::Vec3b(255,128,0),
	cv::Vec3b(255,255,0),
	cv::Vec3b(255,0,0),
	cv::Vec3b(255,0,255),
	cv::Vec3b(255,255,255)
};
	

static void onMouse( int event, int x, int y, int, void* )
{
    if( event != cv::EVENT_LBUTTONDOWN )
		return;
}


void meanDescriptorDistance(const cv::Mat& features, std::vector<float>& meanDistances, int neighbors = 5)
{
	int numFeatures = features.rows;
	//cv::Mat distances = cv::Mat::zeros(numFeatures, numFeatures, CV_32FC1);

	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	std::vector<std::vector<cv::DMatch> > matches;
	meanDistances.resize(numFeatures);

	std::cerr << numFeatures << std::endl;
	matcher->knnMatch(features, features, matches, neighbors+1);
	//std::cerr << matches.size() << std::endl;
	//std::cerr << matches[0].size() << std::endl;

	for (int i = 0; i < numFeatures; i++)
	{
		float sum = 0;
		// Skip the first match, since it's us...
		for (int j = 1; j < neighbors+1; j++)
		{
			cv::DMatch a = matches[i][j];
			sum += a.distance;
		}
		sum /= (neighbors);

		meanDistances[i] = sum;
	}

}

std::string getSystemPath(const std::string packagePath)
{
	std::string filename = "";
	try
	{
		if (packagePath.find("package://") == 0)
		{
			filename = packagePath;
			filename.erase(0, strlen("package://"));
			size_t pos = filename.find("/");
			if (pos != std::string::npos)
			{
				std::string package = filename.substr(0, pos);
				filename.erase(0, pos);
				std::string package_path = ros::package::getPath(package);
				filename = package_path + filename;
			}
		}
		else
		{
			ROS_ERROR("Failed to locate file: %s", packagePath.c_str());
			return filename;
		}
	}
	catch (std::exception& e)
	{
		ROS_ERROR("Failed to retrieve file: %s", e.what());
		return filename;
	}

	return filename;
}

void imageHistogram(const cv::Mat src)
{
	/// Separate the image in 3 places ( B, G and R )
	std::vector<cv::Mat> bgr_planes;
	cv::split( src, bgr_planes );

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

	/// Normalize the result to [ 0, histImage.rows ]
	cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
	cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
	cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
						 cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
						 cv::Scalar( 255, 0, 0), 2, 8, 0  );
		cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
						 cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
						 cv::Scalar( 0, 255, 0), 2, 8, 0  );
		cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
						 cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
						 cv::Scalar( 0, 0, 255), 2, 8, 0  );
	}

	/// Display
	cv::namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE );
	cv::imshow("calcHist Demo", histImage );
}

cv::Mat getKeyPoints(const cv::Mat baseImg, bool useSIFT = false)
{
	std::string featureType = useSIFT ? "SIFT" : "SURF";
	std::vector<cv::KeyPoint> baseKeypoints;
	cv::Mat baseFeatureDescriptors;

	cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create(featureType);
	cv::Ptr<cv::DescriptorExtractor> descriptorExctractor = cv::DescriptorExtractor::create(featureType);

	// Compute search features
	featureDetector->detect(baseImg, baseKeypoints);
	descriptorExctractor->compute(baseImg, baseKeypoints, baseFeatureDescriptors);


	// Compute uniqueness of each feature
	std::vector<float> meanDistances;
	meanDescriptorDistance(baseFeatureDescriptors, meanDistances, 10);
	std::cerr << meanDistances[0] << std::endl;

	cv::Mat outImg = cv::Mat::zeros(baseImg.rows, baseImg.cols, CV_8UC3);
	outImg.setTo(cv::Scalar::all(255));
	std::vector<cv::KeyPoint> tempKeypoint;
//	cv::drawKeypoints(baseImg, baseKeypoints, outImg, keypointColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	cv::imwrite("WhaleFeatures.jpg", outImg);

	const int numLabels = 3;
	cv::Mat labels;
	cv::kmeans(baseFeatureDescriptors, numLabels, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 10, 0.5), 1, cv::KMEANS_PP_CENTERS);

	for (int i = 0; i < meanDistances.size(); i++)
	{
		tempKeypoint.clear();
		tempKeypoint.push_back(baseKeypoints[i]);
		cv::drawKeypoints(outImg, tempKeypoint, outImg, ColorGenerator::jet(labels.at<int>(i)/((float)numLabels-1)), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	}

//	cv::FileStorage fs1 ("/home/arprice/whale_workspace/src/whales/data/features/test_keypoints.xml", cv::FileStorage::WRITE);
//	cv::FileStorage fs2 ("/home/arprice/whale_workspace/src/whales/data/features/test_descriptors.xml", cv::FileStorage::WRITE);
//	fs1 << "keypoints" << cv::Mat(baseKeypoints);
//	fs2 << "descriptors" << baseFeatureDescriptors;

	return outImg;
}

void testMSER(cv::Mat img)
{
	cv::Mat yuv, ellipses;
	cv::cvtColor(img, yuv, cv::COLOR_BGR2YCrCb);
	img.copyTo(ellipses);

	std::vector<std::vector<cv::Point> > contours;
	cv::MSER()(yuv, contours);

	for( int i = (int)contours.size()-1; i >= 0; i-- )
	{
		const std::vector<cv::Point>& r = contours[i];
		for ( int j = 0; j < (int)r.size(); j++ )
		{
			cv::Point pt = r[j];
			ellipses.at<cv::Vec3b>(pt) = bcolors[i%9];
		}

		// find ellipse
		cv::RotatedRect box = cv::fitEllipse( r );
		box.angle = -box.angle;

		box.angle=(float)CV_PI/2-box.angle;
		ellipse( ellipses, box, cv::Scalar(196,255,255), 2 );
	}

	int randNum = rand();
	cv::namedWindow("test" + std::to_string(randNum) + ".jpg", cv::WINDOW_NORMAL);
	cv::imwrite("test" + std::to_string(randNum) + ".jpg", ellipses);
}

std::vector<std::string> enumeratePackageDirectory(const std::string packagePath = "package://whales/data/images/")
{
	std::vector<std::string> filenames;
	boost::filesystem::path full_path(getSystemPath(packagePath));
	if (!boost::filesystem::exists(full_path))
	{
		std::cout << "Unable to find '" << packagePath << "'" << std::endl;
		return filenames;
	}

	if (boost::filesystem::is_directory(full_path))
	{
		std::cout << "Loading directory '" << full_path.string() << "'" << std::endl;
		boost::filesystem::directory_iterator end_iter;
		for (boost::filesystem::directory_iterator dir_iter(full_path);
			 dir_iter != end_iter;
			 ++dir_iter)
		{
			if (boost::filesystem::is_regular_file(dir_iter->status()))
			{
				filenames.push_back(full_path.string() + dir_iter->path().filename().string());
				std::cerr << full_path.string() + dir_iter->path().filename().string() << std::endl;
			}
		}
	}

	return filenames;
}

const bool use_sift = true;
const int neighbors = 1;

int main(int argc, char** argv)
{
	cv::initModule_nonfree();

	// Sets of cv Matrices
	std::vector<cv::Mat> originalImages;
	std::vector<cv::Mat> preprocessedImages;
	std::vector<std::vector<cv::KeyPoint> > keypointSets;
	std::vector<cv::Mat> descriptorSets;

	// Load and preprocess all images
	std::vector<std::string> filenames = enumeratePackageDirectory();
	if (true)
	{
		filenames.clear();
		filenames.push_back("/home/arprice/whale_workspace/src/whales/data/images/WC_0709C.jpg");
		filenames.push_back("/home/arprice/whale_workspace/src/whales/data/images/WC_0709B.jpg");
	}
	const int numFiles = filenames.size();

	originalImages.reserve(numFiles);
	preprocessedImages.reserve(numFiles);
	descriptorSets.reserve(numFiles);
	for (int imgIdx = 0; imgIdx < numFiles; imgIdx++)
	{
		originalImages.push_back(cv::imread(filenames[imgIdx]));

		cv::Mat smoothImage;
		cv::bilateralFilter(originalImages[imgIdx], smoothImage, 10, 80, 25);
		preprocessedImages.push_back(smoothImage);
	}

	std::cout << "Preprocessed all images." << std::endl;

	// Compute SIFT features and descriptors for all images
	std::string featureType = use_sift ? "SIFT" : "SURF";//"FAST"; //"SURF" "MSER";

	cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create("SIFT");
	cv::Ptr<cv::DescriptorExtractor> descriptorExctractor = cv::DescriptorExtractor::create("SIFT");

	for (int imgIdx = 0; imgIdx < numFiles; imgIdx++)
	{
		std::vector<cv::KeyPoint> currentKeypoints;
		cv::Mat currentFeatureDescriptors;

		// Compute search features
		featureDetector->detect(preprocessedImages[imgIdx], currentKeypoints);
		descriptorExctractor->compute(preprocessedImages[imgIdx], currentKeypoints, currentFeatureDescriptors);

		// Store to our database
		keypointSets.push_back(currentKeypoints);
		descriptorSets.push_back(currentFeatureDescriptors);
	}

	// Save results to file for future?

	// Pick a special image and run matching
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	std::vector<std::vector<cv::DMatch> > matches;

	matcher->add(descriptorSets);

	matcher->knnMatch(descriptorSets[0], matches, neighbors+1);

	std::cerr << "Matches: " << matches.size() << std::endl;
	std::cerr << "Matches[0]: " <<  matches[0].size() << std::endl;

	//
	for (int i = 0; i < matches.size(); i++)
	{
		matches[i].erase(matches[i].begin());
	}

	// Get range of feature distances
	float minD = 10000, maxD = 0;
	for (int i = 0; i < matches.size(); i++)
	{
		cv::DMatch a = matches[i][0];
		if (a.distance > maxD) { maxD = a.distance; }
		if (a.distance < minD) { minD = a.distance; }
	}

	std::vector<int> hist; hist.resize(10);
	for (int i = 0; i < matches.size(); i++)
	{
		cv::DMatch a = matches[i][0];
		//std::cerr << a.distance << std::endl;
		if (a.distance > ((maxD-minD)*0.30) + minD)
		{
			matches.erase(matches.begin()+i);
			i--;
		}
		int bin = a.distance/(maxD-minD)*10;
		while (bin >= hist.size()) {hist.push_back(0);}
		hist[bin]++;
	}

	for (int i = 0; i < hist.size(); i++)
	{
		std::cerr << hist[i] << std::endl;
	}

	cv::Mat result;
	cv::drawMatches(originalImages[0], keypointSets[0], originalImages[1], keypointSets[1], matches, result);
	cv::namedWindow("Matches.jpg", cv::WINDOW_NORMAL);
	cv::setMouseCallback("Matches.jpg", onMouse);
	cv::imwrite("Matches.jpg", result);

	testMSER(preprocessedImages[0]);
	testMSER(preprocessedImages[1]);

	cv::namedWindow("Classes.jpg", cv::WINDOW_NORMAL);
	cv::imshow("Classes.jpg", getKeyPoints(preprocessedImages[0], true));

//	cv::Mat image = cv::imread(getSystemPath("package://whales/data/images/0177_11.jpg"));
//	cv::Mat smoothImage;
//	cv::Mat edgeImage;
//	cv::Mat featureImage;
//	cv::Mat segmentedImage;
//	cv::Mat labeledImg;


//	//cv::resize(image, image, cv::Size(1240,960));
//	cv::bilateralFilter(image, smoothImage, 10, 100, 20);
//	cv::Canny(smoothImage, edgeImage, 500, 100, 3);
//	featureImage = getKeyPoints(smoothImage, true);
//	//segment(smoothImage, segmentedImage, labeledImg, 2, 100, 10000);

//	cv::namedWindow("Whale", cv::WINDOW_NORMAL);
//	cv::namedWindow("Filtered", cv::WINDOW_NORMAL);
//	cv::namedWindow("Edges", cv::WINDOW_NORMAL);
//	cv::namedWindow("Features", cv::WINDOW_NORMAL);
//	//cv::namedWindow("Segmented", cv::WINDOW_NORMAL);

//	cv::imshow("Whale", image);
//	cv::imshow("Filtered", smoothImage);
//	cv::imshow("Edges", edgeImage);
//	cv::imshow("Features", featureImage);
	//cv::imshow("Segmented", segmentedImage);

	//imageHistogram(image);

	//cv::imwrite("Whale_Features.jpg", featureImage);

	//while(ros::ok())
	{
		//ros::spinOnce();
		cv::waitKey(0);
	}

	return 0;
}
