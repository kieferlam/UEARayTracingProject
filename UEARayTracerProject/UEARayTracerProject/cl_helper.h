#pragma once

#define CL_TARGET_OPENCL_VERSION 210

#include <CL/opencl.h>
#include <vector>
#include <iostream>
#include <Windows.h>

#define MAX_PLATFORMS (4)
#define PLATFORM_INFO_CHAR_LENGTH (256)

#define BUILD_OPTIONS ("-cl-std=CL2.1")

namespace cl {

	extern cl_platform_id platform;
	extern cl_device_id device;
	extern cl_context context;

	extern cl_program program;

	extern cl_command_queue queue;

	bool init(bool interop);

	void addSource(const std::string& source);

	bool build();

	cl_kernel createKernel(const char* kernelName);

}
