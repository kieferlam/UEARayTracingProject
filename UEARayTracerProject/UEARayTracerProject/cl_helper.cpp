#include "cl_helper.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include "TracerKernel.h"
#include "RARKernel.h"

cl_platform_id retrievePlatform() {
	cl_platform_id platforms[MAX_PLATFORMS];
	cl_uint numPlatforms;
	clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);

	if (numPlatforms == 0) return nullptr;

	// Platform info setup
	const cl_platform_info info_enums[] = {CL_PLATFORM_VERSION, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_PROFILE};

	for (size_t i = 0; i < numPlatforms; ++i) {
		// Loop through enums to retrieve info
		std::cout << "Platform " << i << ":" << std::endl;
		for (int j = 0; j < sizeof(info_enums) / sizeof(info_enums[0]); ++j) {
			std::cout << "\t";
			char info[PLATFORM_INFO_CHAR_LENGTH];
			clGetPlatformInfo(platforms[i], info_enums[j], PLATFORM_INFO_CHAR_LENGTH, info, NULL);
			std::cout << info << std::endl;
		}
	}

	unsigned int platform = numPlatforms;
	if (numPlatforms == 1) platform = 0; // If only one platform, just select that one
	// Ask the user to select a platform.
	while (platform < 0 || platform >= numPlatforms) {
		std::cout << "Select platform by index: ";
		std::cin >> platform;
	}

	return platforms[platform];
}

cl_device_id retrieveDevice(cl_platform_id platform) {
	cl_device_id devices[MAX_PLATFORMS];
	cl_uint numDevices;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAX_PLATFORMS, devices, &numDevices);

	if (numDevices == 0) return nullptr;

	// Device info setup
	const cl_device_info info_enums[] = { CL_DEVICE_NAME, CL_DEVICE_PROFILE, CL_DEVICE_VENDOR, CL_DEVICE_VERSION, CL_DRIVER_VERSION };

	for (size_t i = 0; i < numDevices; ++i) {
		// Loop through enums to retrieve info
		std::cout << "Device " << i << ":" << std::endl;
		for (int j = 0; j < sizeof(info_enums) / sizeof(info_enums[0]); ++j) {
			std::cout << "\t";
			char info[PLATFORM_INFO_CHAR_LENGTH];
			clGetDeviceInfo(devices[i], info_enums[j], PLATFORM_INFO_CHAR_LENGTH, info, NULL);
			std::cout << info << std::endl;
		}
	}

	unsigned int device = numDevices;
	if (numDevices == 1) device = 0; // If only one device, just select that one
	// Ask user to select device
	while (device < 0 || device >= numDevices) {
		std::cout << "Select device by index: ";
		std::cin >> device;
	}

	return devices[device];
}

void loadConfigFromFile(std::string file, std::unordered_map<std::string, std::string>& config) {
	std::ifstream in(file);
	std::string line;
	while (std::getline(in, line)) {
		std::istringstream is_line(line);
		std::string key;
		if (std::getline(is_line, key, '=')) {
			std::string value;
			if (std::getline(is_line, value)) {
				config[key] = value;
			}
		}
	}
}

namespace cl {

	// Externs
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;

	cl_program program;
	cl_command_queue queue;

	device_info_struct device_info;

	std::unordered_map<std::string, std::string> config;

	// Local
	std::vector<std::string> sources;


	void printErrorMsg(std::string msg, int line, const char* filename, cl_int err) {
		if (err == CL_SUCCESS) return;
		std::cout << "Line " << line << " in " << filename << ": " << msg << ": " << cl::getErrorString(err) << std::endl;
	}

	std::string getErrorString(cl_int errorCode) {
		switch (errorCode) {
			// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

			// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

			// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
		}
	}

	std::string getEventString(cl_int eventStatus) {
		switch (eventStatus) {
		case CL_QUEUED:
			return "CL_QUEUED";
		case CL_SUBMITTED:
			return "CL_SUBMITTED";
		case CL_RUNNING:
			return "CL_RUNNING";
		case CL_COMPLETE:
			return "CL_COMPLETE";
		default:
			return "UNKNOWN_EVENT";
		}
	}

	void readEventStatus(cl_event event, cl_int* status) {
		if (event == NULL) return;
		cl_int err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), status, NULL);
	}

	bool init() {
		// Load config
		loadConfigFromFile(CONFIG_FILE, config);

		// CL stuff
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
		if (getConfigBool("useInterop")) {
			context = clCreateContext(props, 1, &device, NULL, NULL, &err);
		} else {
			context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		}

		// Device info
		clGetDeviceInfo(cl::device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_info.local_mem_size), &device_info.local_mem_size, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_info.max_compute_units), &device_info.max_compute_units, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(device_info.max_constant), &device_info.max_constant, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(device_info.max_constant_buffer), &device_info.max_constant_buffer, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(device_info.max_image2d_height), &device_info.max_image2d_height, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(device_info.max_image2d_width), &device_info.max_image2d_width, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(device_info.max_mem_alloc), &device_info.max_mem_alloc, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(device_info.max_parameters), &device_info.max_parameters, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_info.max_work_group_size), &device_info.max_work_group_size, NULL);
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(device_info.max_work_dimensions), &device_info.max_work_dimensions, NULL);
		device_info.max_work_item_sizes = new size_t[device_info.max_work_dimensions];
		clGetDeviceInfo(cl::device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * device_info.max_work_dimensions, device_info.max_work_item_sizes, NULL);

		std::cout << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_MAX_COMPUTE_UNITS: " << std::setw(8) << device_info.max_compute_units << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << std::setw(8) << device_info.max_work_group_size << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << std::setw(8) << device_info.max_work_dimensions << std::endl;

		std::cout << std::setw(48) << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << std::setw(8);
		for (size_t i = 0; i < device_info.max_work_dimensions; ++i) std::cout << device_info.max_work_item_sizes[i] << ",";
		std::cout << std::endl;

		std::cout << std::setw(48) << "CL_DEVICE_MAX_CONSTANT_ARGS: " << std::setw(8) << device_info.max_constant << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_LOCAL_MEM_SIZE: " << std::setw(8) << device_info.local_mem_size << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << std::setw(8) << device_info.max_constant_buffer << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << std::setw(8) << device_info.max_mem_alloc << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_MAX_PARAMETER_SIZE: " << std::setw(8) << device_info.max_parameters << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_IMAGE2D_MAX_WIDTH: " << std::setw(8) << device_info.max_image2d_width << std::endl;
		std::cout << std::setw(48) << "CL_DEVICE_IMAGE2D_MAX_HEIGHT: " << std::setw(8) << device_info.max_image2d_height << std::endl;

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

	void addSource(const std::string source) {
		sources.push_back(source);
	}

	std::string getBuildOptions() {
		std::ostringstream stream;
		stream << BUILD_OPTIONS
			<< " -D GRID_CELL_ROW_COUNT=" << GRID_CELL_ROW_COUNT
			<< " -D GRID_MAX_TRIANGLES_PER_CELL=" << GRID_MAX_TRIANGLES_PER_CELL
			<< " -D NUM_RAY_CHILDREN=" << NUM_RAY_CHILDREN
			<< " -g "; 
		if (getConfigBool("useInterop")) stream << "-D USE_INTEROP ";
		if (getConfigBool("disableWarnings")) stream << "-w ";
		if (getConfigBool("makeWarningsErrors")) stream << "-Werror ";
		if (getConfigBool("disableOptimisation")) stream << "-cl-opt-disable ";
		if (getConfigBool("enableMad")) stream << "-cl-mad-enable ";
		if (getConfigBool("enableUnsafeMaths")) stream << "-cl-unsafe-math-optimizations ";
		if (getConfigBool("finiteMathsOnly")) stream << "-cl-finite-math-only ";
		if (getConfigBool("fastRelaxedMaths")) stream << "-cl-fast-relaxed-math ";

		if (config.find("buildOptions") != config.end()) {
			stream << config["buildOptions"] << " ";
		}

		return stream.str();
	}

	bool build() {
		cl_int err;

		std::vector<const char*> srcvec;
		std::vector<size_t> srclen;
		for (auto it = sources.begin(); it != sources.end(); ++it) {
			srcvec.push_back(it->c_str());
			srclen.push_back(it->length());
		}

 		program = clCreateProgramWithSource(context, sources.size(), &srcvec[0], &srclen[0], &err);
		if (err != NULL) {
			std::cout << "Program creation error: " << getErrorString(err) << std::endl;
		}

		std::string buildOptions = getBuildOptions();
		std::cout << "BuildOptions: " << buildOptions << std::endl;
		err = clBuildProgram(program, 1, &device, buildOptions.c_str(), NULL, NULL);
		if (err == CL_SUCCESS) {
			queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
			sources.clear(); // Deallocate sources
			if (err == NULL) return true;
			std::cout << "Could not create command queue: " << getErrorString(err) << std::endl;
			return false;
		}

		// IF BUILD FAILS
		std::cout << "Build error: " << getErrorString(err) << std::endl;

		size_t buildLogLength;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &buildLogLength);
		char* buildLog = new char[buildLogLength];
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, buildLogLength, buildLog, NULL);
		if (err != CL_SUCCESS) {
			std::cout << "Failed to get program build log: " << getErrorString(err) << std::endl;
		} else {
			std::cout << buildLog << std::endl;
		}
		delete[] buildLog;

		return false;
	}

	cl_kernel createKernel(const char* kernelName) {
		cl_int err;
		cl_kernel kernel = clCreateKernel(program, kernelName, &err);
		if (err == NULL) return kernel;
		std::cout << "Kernel creation error on '" << kernelName << "': " << getErrorString(err) << std::endl;
		return nullptr;
	}

	bool getConfigBool(std::string key) {
		if (config.find(key) == config.end()) return false;
		return config[key] == "true";
	}

	int getConfigInt(std::string key) {
		if (config.find(key) == config.end()) return -1;
		return std::stoi(config[key]);
	}
	
	float getConfigFloat(std::string key) {
		if (config.find(key) == config.end()) return -1;
		return std::stof(config[key]);
	}
}
