/**
 * \file common.cpp
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

#include "whales/common.h"

#include <iostream>
#include <fstream>
#include <streambuf>

#include <ros/ros.h>
#include <ros/package.h>

#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace whales
{

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

std::vector<std::string> enumeratePackageDirectory(const std::string packagePath)
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

}
