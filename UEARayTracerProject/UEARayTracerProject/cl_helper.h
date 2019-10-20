#pragma once

#include <CL/opencl.h>
#include <vector>
#include <iostream>
#include <Windows.h>

#define MAX_PLATFORMS (4)
#define PLATFORM_INFO_CHAR_LENGTH (256)

namespace cl {

	extern cl_platform_id platform;
	extern cl_device_id device;
	extern cl_context context;

	bool init(bool interop);
}
