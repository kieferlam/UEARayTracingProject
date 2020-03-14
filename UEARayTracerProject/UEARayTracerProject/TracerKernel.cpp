#include "TracerKernel.h"
#include <SOIL.h>

void _loadSkyboxTexture(unsigned char** skyboxImages, int* width, int* height) {

	// Load skybox textures
	std::string skyboxFilePrefix = "data/skybox/";
	std::string skyboxFiles[] = { skyboxFilePrefix + "right.png", skyboxFilePrefix + "left.png", skyboxFilePrefix + "up.png", skyboxFilePrefix + "down.png", skyboxFilePrefix + "back.png", skyboxFilePrefix + "front.png" };

	for (int i = 0; i < 6; ++i) {
		skyboxImages[i] = SOIL_load_image(skyboxFiles[i].c_str(), width, height, NULL, SOIL_LOAD_RGB);
		if (skyboxImages[i] == nullptr) {
			std::cout << "Could not load skybox cubemap: " << SOIL_last_result() << std::endl;
		}
	}
}

bool TracerKernel::create(GLuint texture, int width, int height) {
	// Create the kernel
	kernel = cl::createKernel(kernelName.c_str());
	if (kernel == nullptr) {
		std::cout << "Could not create kernel." << std::endl;
		return false;
	}

	// Load skybox texture
	int skyboxWidth;
	int skyboxHeight;
	_loadSkyboxTexture(skyboxImages, &skyboxWidth, &skyboxHeight);

	// Create memory buffers
	
	// Kernel input parameter
	cl_int err;
	kernelInputStruct.aspect = (float)width / (float)height;
	kernelInputStruct.width = width;
	kernelInputStruct.height = height;
	kernelInputStruct.screenDistance = 2.0f;
	kernelInputStruct.camera = { 0.0f, 0.0f, 0.0f };
	inputKernelBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(KernelInputStruct), &kernelInputStruct, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error creating kernel parameter buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Image output buffer
	outputImageBuffer = clCreateFromGLTexture(cl::context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error creating kernel image output buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Config buffer
	config.skyboxSize = { skyboxWidth, skyboxHeight };
	configBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(RTConfig), &config, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error creating rt config buffer: " << cl::getErrorString(err) << std::endl;
	}

	// Skybox buffers
	size_t rgbsize = (size_t)skyboxWidth * skyboxHeight * 3;
	unsigned char* skbuf = new unsigned char[rgbsize * 6];
	for (int i = 0; i < 6; ++i) {
		memcpy(skbuf + (i * rgbsize), skyboxImages[i], rgbsize * sizeof(unsigned char));
	}
	skyboxBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 6 * rgbsize * sizeof(unsigned char), skbuf, &err);
	if (err != CL_SUCCESS) {
		std::cout << "Error creating skybox buffer: " << cl::getErrorString(err) << std::endl;
	}
	delete[] skbuf;

	// Set kernel parameters
	err = clSetKernelArg(kernel, 0, sizeof(outputImageBuffer), &outputImageBuffer);
	if (err != CL_SUCCESS) std::cout << "Couldn't set output kernel arg: " << cl::getErrorString(err) << std::endl;
	err = clSetKernelArg(kernel, 1, sizeof(inputKernelBuffer), &inputKernelBuffer);
	if (err != CL_SUCCESS) std::cout << "Couldn't set input kernel arg: " << cl::getErrorString(err) << std::endl;
	err = clSetKernelArg(kernel, 2, sizeof(configBuffer), &configBuffer);
	if (err != CL_SUCCESS) std::cout << "Couldn't set config kernel arg: " << cl::getErrorString(err) << std::endl;
	err = clSetKernelArg(kernel, 3, sizeof(skyboxBuffer), &skyboxBuffer);
	if (err != CL_SUCCESS) std::cout << "Couldn't set skybox kernel arg: " << cl::getErrorString(err) << std::endl;

	worksize[0] = width;
	worksize[1] = height;

	// data
	srand(time(NULL));


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

void TracerKernel::keyevent(int key, int scancode, int action, int mods) {
}
