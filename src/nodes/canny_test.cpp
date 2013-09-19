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

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "fh_segmentation/fh_segmentation.h"


const cv::Scalar keypointColor = cv::Scalar(0, 255, 0); // Green Keypoints
const std::string modelName = "boxModel";
//const std::string modelString = "<?xml version='1.0'?><sdf version='1.4'><model name=\"my_robot\"><static>false</static><link name='link'><pose>0 0 0 0 0 0</pose><collision name='collision'><geometry><box><size>1 .1 .05</size></box></geometry></collision><visual name='visual'><geometry><box><size>1 .1 .05</size></box></geometry></visual></link></model></sdf>";

std::vector<std::string> objectNames;

ros::Publisher zPub;

ros::ServiceClient worldInfoClient;
ros::ServiceClient spawnClient;
ros::ServiceClient deleteClient;

float randbetween(float min, float max)
{
	return (max - min) * ( (double)rand() / (double)RAND_MAX ) + min;
}

cv::Mat loadModelSDF(const std::string file)
{
	cv::Mat outImg;
	std::string filename;
	try
	{
		if (file.find("package://") == 0)
		{
			filename = file;
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
			ROS_ERROR("Failed to locate file: %s", file.c_str());
			return outImg;
		}
	}
	catch (std::exception& e)
	{
		ROS_ERROR("Failed to retrieve file: %s", e.what());
		return outImg;
	}



	outImg = cv::imread(filename);

	return outImg;
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

	cv::Mat outImg;
	cv::drawKeypoints(baseImg, baseKeypoints, outImg, keypointColor, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	return outImg;
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "random_spawn");
	ros::NodeHandle nh;

	cv::initModule_nonfree();


	cv::Mat image = loadModelSDF("package://whales/data/images/Mn020.jpg");
	cv::Mat smoothImage;
	cv::Mat edgeImage;
	cv::Mat featureImage;
	cv::Mat segmentedImage;
	cv::Mat labeledImg;


	//cv::resize(image, image, cv::Size(1240,960));
	cv::bilateralFilter(image, smoothImage, 10, 100, 20);
	cv::Canny(smoothImage, edgeImage, 500, 100, 3);
	featureImage = getKeyPoints(smoothImage, true);
	segment(smoothImage, segmentedImage, labeledImg, 2, 100, 10000);

	cv::namedWindow("Whale", cv::WINDOW_NORMAL);
	cv::namedWindow("Filtered", cv::WINDOW_NORMAL);
	cv::namedWindow("Edges", cv::WINDOW_NORMAL);
	cv::namedWindow("Features", cv::WINDOW_NORMAL);
	cv::namedWindow("Segmented", cv::WINDOW_NORMAL);
	//cv::namedWindow("Whale");

	cv::imshow("Whale", image);
	cv::imshow("Filtered", smoothImage);
	cv::imshow("Edges", edgeImage);
	cv::imshow("Features", featureImage);
	cv::imshow("Segmented", segmentedImage);

	//imageHistogram(image);

	//cv::imwrite("Whale_Features.jpg", featureImage);

	while(ros::ok())
	{
		ros::spinOnce();
		cv::waitKey(1);
	}

	return 0;
}
