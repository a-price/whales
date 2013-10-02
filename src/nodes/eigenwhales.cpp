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

void findExtrema(const cv::Mat img)
{
	int scale = 1;
	int delta = 0;
	int ksize = 5;
	int size = 3;
	int ddepth = CV_32F;

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
	/// Normalizing
	cv::normalize( corners, corners, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
	cv::convertScaleAbs( corners, corners );

	smooth.convertTo(smooth, CV_8U);
	cv::Canny(smooth, edge, 5, 5*3, 5);
	cv::normalize(edge, edge, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	//cv::cvtColor(result, result, CV_BGR2HSV);
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


			channels[0].at<float>(idx) = (atan2(dy, dx) + M_PI) * 180.0/M_2_PI;
			channels[1].at<float>(idx) = sqrt(dx * dx + dy * dy);
			channels[2].at<float>(idx) = 255;
			//cv::Vec3b pixel(((atan2(dy, dx) + M_PI) * 180.0/M_2_PI), sqrt(dx * dx + dy * dy), 255);

			//result.at<cv::Vec3b>(idx) = pixel;
		}
	}

	cv::normalize(channels[1], channels[1], 0, 255, cv::NORM_MINMAX, CV_32FC1);
	channels[0].convertTo(channels[0], CV_8U);
	channels[1].convertTo(channels[1], CV_8U);
	channels[2] = corners; channels[2].convertTo(channels[2], CV_8U);
	cv::merge(channels, result);

	cv::cvtColor(result, result, CV_HSV2BGR);

	cv::normalize(magnitude, magnitude, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
	cv::normalize(orientation, orientation, 0, 1.0, cv::NORM_MINMAX, CV_32FC1);
	std::cerr << "Hello." << std::endl;
	//cv::normalize(result, result, 0, 180, cv::NORM_MINMAX, CV_8UC3);

	cv::imshow("Hello", result);
}

int main(int argc, char** argv)
{
	cv::namedWindow("Hello", cv::WINDOW_NORMAL);
	// Data storage
	std::vector<cv::Mat> originalImages;

	// Load a database of images
	std::vector<std::string> filenames = enumeratePackageDirectory("package://whales/data/images/eigen/");
	const int numFiles = filenames.size();
	originalImages.reserve(numFiles);
	for (int imgIdx = 0; imgIdx < numFiles; imgIdx++)
	{
		cv::Mat temp = cv::imread(filenames[imgIdx], CV_LOAD_IMAGE_GRAYSCALE);
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
