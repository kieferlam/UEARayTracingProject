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
	kernelInputStruct.aspect = (float)width / (float)height;
	kernelInputStruct.width = width;
	kernelInputStruct.height = height;
	kernelInputStruct.screenDistance = 1.0f;
	kernelInputStruct.camera = { 0.0f, 0.0f, 0.0f };
	inputKernelBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(KernelInputStruct), &kernelInputStruct, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error creating kernel parameter buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Image output buffer
	outputImageBuffer = clCreateFromGLTexture(cl::context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, texture, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error creating kernel image output buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Set kernel parameters
	err = clSetKernelArg(kernel, 0, sizeof(outputImageBuffer), &outputImageBuffer);
	if (err != CL_SUCCESS) std::cout << "Couldn't set output kernel arg: " << cl::getErrorString(err) << std::endl;
	err = clSetKernelArg(kernel, 1, sizeof(inputKernelBuffer), &inputKernelBuffer);
	if (err != CL_SUCCESS) std::cout << "Couldn't set input kernel arg: " << cl::getErrorString(err) << std::endl;

	worksize[0] = width;
	worksize[1] = height;

	// data
	srand(time(NULL));

	// Meta

	// Spheres
	for (int i = 0; i < 1; ++i) {
		kernelInputStruct.spheres[i].position = { 0.0f, 0.0f, 20.0f};
		kernelInputStruct.spheres[i].colour = { 1.0f, 0.8f, 0.9f };
		kernelInputStruct.spheres[i].radius = 10.0f;
	}
	kernelInputStruct.numSpheres = 1;

	return true;
}

void TracerKernel::writeKernelInput(cl_bool block) {
	cl_int error = clEnqueueWriteBuffer(cl::queue, inputKernelBuffer, block, 0, sizeof(KernelInputStruct), &kernelInputStruct, 0, NULL, NULL);
	if (error != CL_SUCCESS) std::cout << "Error writing kernel input: " << cl::getErrorString(error) << std::endl;
}

void TracerKernel::trace(cl_bool block) {
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
