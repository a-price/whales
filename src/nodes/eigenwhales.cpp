/**
 * \file eigenwhales.cpp
 * \brief
 *
 * \author Andrew Price
 * \date 9 30, 2013
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

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#include "whales/common.h"

const int INV_SCALE = 4;
const int WIDTH = 640/INV_SCALE;
const int HEIGHT = 480/INV_SCALE;
const int LENGTH = WIDTH * HEIGHT;

using namespace whales;
using namespace Eigen;

typedef  Map<Matrix<float, 1, LENGTH> > VectorImage;

inline VectorImage vectorize(const cv::Mat img)
{
	return VectorImage((float*)img.data);
}

double wrapTo2Pi(double x)
{
	x = fmod(x, 2*M_PI);
	if (x < 0)
		x += 2*M_PI;
	return x;
}

inline double angleBetween(double a, double b)
{
	return fmod((b-a) + M_PI, 2*M_PI) - M_PI;
}

float pointScore(const float magnitude, const float orientation, const cv::Point& location, const float targetOrientation, const cv::Point& targetLocation)
{
	float dx = targetLocation.x - location.x;
	float dy = targetLocation.y - location.y;
	float d = sqrt(dx*dx + dy*dy);

	float dTheta = angleBetween(wrapTo2Pi(targetOrientation), wrapTo2Pi(orientation));

	return sqrt(magnitude) / (sqrt(d+0.1) * (fabs(dTheta) + 0.1));
}

void findExtrema(const cv::Mat img)
{
	int scale = 1;
	int delta = 0;
	int ksize = 5;
	int size = 3;
	int ddepth = CV_32F;

	const cv::Point targetA(5, img.rows/3);
	const cv::Point targetB(img.cols/2, img.rows/2);
	const cv::Point targetC(img.cols - 1 - 5, img.rows/3);

	const cv::Point targetD(img.cols/3, 2*img.rows/3);
	const cv::Point targetE(2*img.cols/3, 2*img.rows/3);

	cv::Mat smooth;
	cv::Mat edge;
	cv::Mat corners = cv::Mat::zeros(img.size(), CV_32FC1);
	cv::Mat grad_x, grad_y;
	cv::Mat orientation = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
	cv::Mat magnitude = cv::Mat::zeros(img.rows, img.cols, CV_32FC1);
	cv::Mat result = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);

	cv::GaussianBlur(img, smooth, cv::Size(size, size), 0, 0, cv::BORDER_DEFAULT);

	/// Gradient X
	cv::Sobel( smooth, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT );
	/// Gradient Y
	cv::Sobel( smooth, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT );

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = ksize;
	double k = 0.04;

	/// Detecting corners
	cv::cornerHarris( smooth, corners, blockSize, apertureSize, k, cv::BORDER_DEFAULT );


//	smooth.convertTo(smooth, CV_8U);
//	cv::Canny(smooth, edge, 5, 5*3, 5);
//	cv::normalize(edge, edge, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	std::vector<cv::Mat> channels;
	cv::split(result, channels);


	for (int v = 0; v < img.rows; v++)
	{
		for (int u = 0; u < img.cols; u++)
		{
			int idx = (v * img.cols) + u;
			float dx = grad_x.at<float>(idx);
			float dy = grad_y.at<float>(idx);
			orientation.at<float>(idx) = atan2(dy, dx);
			magnitude.at<float>(idx) = sqrt(dx * dx + dy * dy);

			channels[0].at<float>(idx) = (atan2(dy, dx) + M_PI) * 180.0/(2*M_PI);
			channels[1].at<float>(idx) = sqrt(dx * dx + dy * dy);
		}
	}

	// Find max in each 1/3rd
	float maxA = 0, maxB = 0, maxC = 0, maxD = 0, maxE = 0;
	cv::Point idxA, idxB, idxC, idxD, idxE;
	for (int v = 0; v < img.rows; v++)
	{
		for (int u = 0; u < img.cols; u++)
		{
			int idx = (v * img.cols) + u;
			float intensity = fabs(corners.at<float>(idx));
			float localOrientation = channels[0].at<float>(idx);
			if (u < img.cols/4.0)
			{
				float score = pointScore(intensity, localOrientation, cv::Point(u,v), 0.0, targetA);
				if (score > maxA) { maxA = score; idxA = cv::Point(u,v); }
			}
			else if (u < 3.0*img.cols/7.0)
			{
				float score = pointScore(intensity, localOrientation, cv::Point(u,v), M_PI/4.0, targetD);
				if (score > maxD) { maxD = score; idxD = cv::Point(u,v); }
			}
			else if (u < 4.0*img.cols/7.0)
			{
				float score = pointScore(intensity, localOrientation, cv::Point(u,v), M_PI/4.0, targetB);
				if (score > maxB) { maxB = score; idxB = cv::Point(u,v); }
			}
			else if (u < 3.0*img.cols/4.0)
			{
				float score = pointScore(intensity, localOrientation, cv::Point(u,v), M_PI/4.0, targetE);
				if (score > maxE) { maxE = score; idxE = cv::Point(u,v); }
			}
			else
			{
				float score = pointScore(intensity, localOrientation, cv::Point(u,v), 0.0, targetC);
				if (score > maxC) { maxC = score; idxC = cv::Point(u,v); }
			}
		}

	}
	/// Normalizing
	cv::normalize( corners, corners, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
	cv::convertScaleAbs( corners, corners );


	// Reassemble HSV channels
	cv::normalize(channels[1], channels[1], 0, 255, cv::NORM_MINMAX, CV_32FC1);
	channels[0].convertTo(channels[0], CV_8U);
	channels[1].convertTo(channels[1], CV_8U);
	channels[2] = corners; channels[2].convertTo(channels[2], CV_8U);
	cv::merge(channels, result);

	cv::cvtColor(result, result, CV_HSV2BGR);

	// Circle targets
	cv::circle(result, targetA, 4, cv::Scalar(0, 0, 255));
	cv::circle(result, targetB, 4, cv::Scalar(0, 0, 255));
	cv::circle(result, targetC, 4, cv::Scalar(0, 0, 255));
	cv::circle(result, targetD, 4, cv::Scalar(0, 0, 255));
	cv::circle(result, targetE, 4, cv::Scalar(0, 0, 255));
	// Circle keypoints
	cv::circle(result, idxA, 4, cv::Scalar::all(255));
	cv::circle(result, idxB, 4, cv::Scalar::all(255));
	cv::circle(result, idxC, 4, cv::Scalar::all(255));
	cv::circle(result, idxD, 4, cv::Scalar::all(255));
	cv::circle(result, idxE, 4, cv::Scalar::all(255));

	cv::normalize(magnitude, magnitude, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
	cv::normalize(orientation, orientation, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);

	cv::imshow("Hello", result);
}

int main(int argc, char** argv)
{
	cv::namedWindow("Hello", cv::WINDOW_NORMAL);
	// Data storage
	std::vector<cv::Mat> originalImages;

	// Load a database of images
	std::vector<std::string> filenames = enumeratePackageDirectory("package://whales/data/images/eigen/");
	int numFiles = filenames.size();
	originalImages.reserve(numFiles);
	int numFails = 0;
	for (int imgIdx = 0; imgIdx < numFiles; imgIdx++)
	{
		cv::Mat temp = cv::imread(filenames[imgIdx], CV_LOAD_IMAGE_GRAYSCALE);
		if (!temp.data) { numFails++; continue;}
		// Resize
		cv::resize(temp, temp, cv::Size(WIDTH, HEIGHT));
		temp.convertTo(temp, CV_32F);
		temp /= 255.0;
		// Equalize?
		cv::normalize(temp, temp, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);

		if (imgIdx > 1 && imgIdx < 20)
		{
			findExtrema(temp);
			cv::waitKey(0);
		}
		// Store
		originalImages.push_back(temp);
	}
	//cv::imshow("img", originalImages[0]);
	return 0;

	numFiles -= numFails;
	std::cerr << "Number of Images: " << numFiles << std::endl;

	// Vectorize
	VectorImage test = vectorize(originalImages[0]);
	MatrixXf A(numFiles, LENGTH);
	for (int imgIdx = 0; imgIdx < numFiles; imgIdx++)
	{
		A.row(imgIdx) = vectorize(originalImages[imgIdx]);
	}
	std::cerr << A(5,5) << std::endl;

	// PCA
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
	Eigen::MatrixXf basis = svd.matrixV();
	std::cerr << basis.rows() << "x" << basis.cols() << std::endl;

	cv::Mat1f best = cv::Mat1f::zeros(cv::Size(WIDTH, HEIGHT));
	VectorImage alias = vectorize(best);

	std::vector<cv::Mat> bases;

	for(int i = 0; i < 10; i++)
	{
		alias = -basis.col(i).transpose();
		cv::normalize(best, best, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
		//bases.push_back(best.copyTo(););
		cv::imshow("basis" + std::to_string(i), best);
	}

	cv::waitKey(0);
	return 0;
}
