#include "cl_helper.h"


std::string getErrorString(cl_int errorCode) {
	if (errorCode == NULL) return "NO_ERROR";
	const cl_int errCodes[] = {
		CL_INVALID_CONTEXT,
		CL_INVALID_VALUE,
		CL_OUT_OF_HOST_MEMORY,
		CL_INVALID_PROGRAM,
		CL_INVALID_DEVICE,
		CL_INVALID_BINARY,
		CL_INVALID_BUILD_OPTIONS,
		CL_INVALID_OPERATION,
		CL_COMPILER_NOT_AVAILABLE,
		CL_BUILD_PROGRAM_FAILURE,
		CL_INVALID_QUEUE_PROPERTIES,
		CL_OUT_OF_RESOURCES,
		CL_INVALID_PROGRAM_EXECUTABLE,
		CL_INVALID_KERNEL_NAME,
		CL_INVALID_KERNEL_DEFINITION
	};
	const std::string errStrings[] = {
		"CL_INVALID_CONTEXT",
		"CL_INVALID_VALUE",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_OPERATION",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_OUT_OF_RESOURCES",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION"
	};
	for (int i = 0; i < sizeof(errCodes) / sizeof(errCodes[0]); ++i) {
		if (errCodes[i] == errorCode) return errStrings[i];
	}
	return "UNKNOWN_ERROR_CODE";
}

cl_platform_id retrievePlatform() {
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_uint numPlatforms;
	clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);

	if (numPlatforms == 0) return nullptr;

	// Platform info setup
	const size_t info_enums[] = {CL_PLATFORM_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_PROFILE};

	for (int i = 0; i < numPlatforms; ++i) {
		// Loop through enums to retrieve info
		std::cout << "Platform " << i << ":" << std::endl;
		for (int j = 0; j < sizeof(info_enums) / sizeof(info_enums[0]); ++j) {
			std::cout << "\t";
			char info[PLATFORM_INFO_CHAR_LENGTH];
			clGetPlatformInfo(platforms[i], info_enums[j], PLATFORM_INFO_CHAR_LENGTH, info, NULL);
			std::cout << info << std::endl;
		}
	}

	return platforms[numPlatforms - 1];
}

cl_device_id retrieveDevice(cl_platform_id platform) {
	cl_device_id devices[MAX_PLATFORMS];
	cl_uint numDevices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAX_PLATFORMS, devices, &numDevices);

	if (numDevices == 0) return nullptr;

	// Device info setup
	const size_t info_enums[] = { CL_DEVICE_NAME, CL_DEVICE_PROFILE, CL_DEVICE_VENDOR, CL_DEVICE_VERSION, CL_DRIVER_VERSION };

	for (int i = 0; i < numDevices; ++i) {
		// Loop through enums to retrieve info
		std::cout << "Device " << i << ":" << std::endl;
		for (int j = 0; j < sizeof(info_enums) / sizeof(info_enums[0]); ++j) {
			std::cout << "\t";
			char info[PLATFORM_INFO_CHAR_LENGTH];
			clGetDeviceInfo(devices[i], info_enums[j], PLATFORM_INFO_CHAR_LENGTH, info, NULL);
			std::cout << info << std::endl;
		}
	}

	return devices[numDevices - 1];
}

namespace cl {

	// Externs
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;

	cl_program program;
	cl_command_queue queue;

	// Local
	std::vector<const char*> sources;
	std::vector<size_t> sourceLengths;

	bool init(bool interop) {
		platform = retrievePlatform();
		if (platform == nullptr) {
			std::cout << "Could not retrieve platform." << std::endl;
			return false;
		}

		device = retrieveDevice(platform);
		if (device == nullptr) {
			std::cout << "Could not retrieve device." << std::endl;
			return false;
		}

		// Check extensions
		char extensions[2048];
		clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 2048, extensions, NULL);
		std::cout << extensions << std::endl;

		// Create context
		cl_context_properties props[7] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties) wglGetCurrentContext(),
			CL_WGL_HDC_KHR, (cl_context_properties) wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
			0
		};

		cl_int err;
		if (interop) {
			context = clCreateContext(props, 1, &device, NULL, NULL, &err);
		} else {
			context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		}

		if (err == NULL) return true;

		// Error reporting
		cl_int errCodes[] = { CL_INVALID_PLATFORM, CL_INVALID_VALUE, CL_INVALID_DEVICE, CL_DEVICE_NOT_AVAILABLE, CL_OUT_OF_HOST_MEMORY };
		const char* errStrings[] = { "CL_INVALID_PLATFORM", "CL_INVALID_VALUE", "CL_INVALID_DEVICE", "CL_DEVICE_NOT_AVAILABLE", "CL_OUT_OF_HOST_MEMORY" };

		for (int i = 0; i < sizeof(errCodes) / sizeof(errCodes[0]); ++i) {
			if (err == errCodes[i]) {
				std::cout << "Context creation error: " << errStrings[i] << std::endl;
			}
		}

		return false;

	}

	void addSource(const std::string& source) {
		sources.push_back(source.c_str());
		sourceLengths.push_back(source.length());
	}

	bool build() {
		cl_int err;
		program = clCreateProgramWithSource(context, sources.size(), &sources[0], &sourceLengths[0], &err);
		if (err != NULL) {
			std::cout << "Program creation error: " << getErrorString(err) << std::endl;
		}

		err = clBuildProgram(program, 1, &device, BUILD_OPTIONS, NULL, NULL);
		if (err == CL_SUCCESS) {
			queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
			if (err == NULL) return true;
			std::cout << "Could not create command queue: " << getErrorString(err) << std::endl;
			return false;
		}

		// IF BUILD FAILS
		std::cout << "Build error: " << getErrorString(err) << std::endl;

		char buildLog[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 2048, buildLog, NULL);
		std::cout << buildLog << std::endl;

		return false;
	}

	cl_kernel createKernel(const char* kernelName) {
		cl_int err;
		cl_kernel kernel = clCreateKernel(program, kernelName, &err);
		if (err == NULL) return kernel;
		std::cout << "Kernel creation error on '" << kernelName << "': " << getErrorString(err) << std::endl;
		return nullptr;
	}
}
