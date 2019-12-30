#pragma once

#define CL_TARGET_OPENCL_VERSION 210

#include <CL/opencl.h>
#include <vector>
#include <iostream>
#include <Windows.h>
#include <string>
#include <unordered_map>

#define CONFIG_FILE ("config.ini")

#define MAX_PLATFORMS (4)
#define PLATFORM_INFO_CHAR_LENGTH (256)

#define BUILD_OPTIONS ("-cl-std=CL2.1")

namespace cl {

	extern cl_platform_id platform;
	extern cl_device_id device;
	extern cl_context context;

	extern cl_program program;

	extern cl_command_queue queue;

	extern std::unordered_map<std::string, std::string> config;

	std::string getErrorString(cl_int errorCode);

	bool init();

	void addSource(const std::string& source);

	bool build();

	cl_kernel createKernel(const char* kernelName);

	bool getConfigBool(std::string key);

	int getConfigInt(std::string key);

	float getConfigFloat(std::string key);
}
