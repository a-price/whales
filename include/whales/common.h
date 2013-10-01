/**
 * \file common.h
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

#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace whales
{

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

std::string getSystemPath(const std::string packagePath);
std::vector<std::string> enumeratePackageDirectory(const std::string packagePath = "package://whales/data/images/");

void subplot(int rows, int cols, int imgWidth, int imgHeight, cv::Mat currentImage, const cv::Mat newImage);

}

#endif // COMMON_H
