#include "TracerKernel.h"

bool TracerKernel::create(GLuint texture) {
	// Create the kernel
	kernel = cl::createKernel(kernelName.c_str());
	if (kernel == nullptr) {
		std::cout << "Could not create kernel." << std::endl;
		return false;
	}

	// Create memory buffers
	
	// Kernel input parameter
	cl_int err;
	paramInput = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(kernelInputStruct), &kernelInputStruct, &err);

	return true;
}
