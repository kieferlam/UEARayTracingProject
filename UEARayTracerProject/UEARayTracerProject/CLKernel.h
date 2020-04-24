#pragma once
#include <CL/opencl.h>
#include <string>
#include <iostream>
#include "cl_helper.h"
#include "World.h"

class CLKernel {

	const std::string kernelName;

	cl_kernel kernel;

	void display_kernel_info();

public:
	inline CLKernel() : kernelName("INVALID") { };
	inline CLKernel(std::string name) : kernelName(name) { kernel = nullptr; };
	inline ~CLKernel() {}

	inline std::string getKernelName() { return kernelName; }

	inline cl_kernel getKernel() { return kernel; }

	inline bool createKernel() {
		kernel = cl::createKernel(getKernelName().c_str());
		if (kernel == nullptr) {
			std::cout << "Could not create " << getKernelName() << " kernel." << std::endl;
			return false;
		}
		display_kernel_info();
		return true;
	}

	virtual void create() = 0;

	virtual cl_event update() = 0;

	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) = 0;

	virtual void destroy() = 0;
};

