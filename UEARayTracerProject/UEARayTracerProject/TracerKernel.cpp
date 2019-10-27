#include "TracerKernel.h"

bool TracerKernel::create(GLuint texture, int width, int height) {
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
	if (err != NULL) {
		std::cout << "Error creating kernel parameter buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Image output buffer
	outputImageBuffer = clCreateFromGLTexture(cl::context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, texture, &err);
	if (err != NULL) {
		std::cout << "Error creating kernel image output buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Set kernel parameters
	clSetKernelArg(kernel, 0, sizeof(outputImageBuffer), &outputImageBuffer);
	clSetKernelArg(kernel, 1, sizeof(paramInput), &paramInput);

	worksize[0] = width;
	worksize[1] = height;

	return true;
}

void TracerKernel::trace(bool block) {
	cl_int error;

	// Acquire OpenGL texture
	error = clEnqueueAcquireGLObjects(cl::queue, 1, &outputImageBuffer, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error acquiring GL objects: " << cl::getErrorString(error) << std::endl;

	// Queue tracer
	error = clEnqueueNDRangeKernel(cl::queue, kernel, 2, 0, worksize, NULL, NULL, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error queueing kernel: " << cl::getErrorString(error) << std::endl;

	// Release OpenGL texture
	error = clEnqueueReleaseGLObjects(cl::queue, 1, &outputImageBuffer, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error release GL objects: " << cl::getErrorString(error) << std::endl;

	if(block) clFinish(cl::queue);
}
