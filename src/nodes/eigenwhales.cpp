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

int main(int argc, char** argv)
{
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
		// Store
		originalImages.push_back(temp);
	}
	//cv::imshow("img", originalImages[0]);

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
