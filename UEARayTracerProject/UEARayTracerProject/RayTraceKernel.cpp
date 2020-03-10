#include "RayTraceKernel.h"
#include <SOIL.h>

void loadSkyboxTexture(unsigned char** skyboxImages, int* width, int* height) {

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

RayTraceKernel::RayTraceKernel() : CLKernel("RayTrace") {
}

RayTraceKernel::~RayTraceKernel() {
}

void RayTraceKernel::create() {

	// Assume kernel object has been created

	cl_int err;

	configBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(*config), config, &err);
	cl::printErrorMsg("Config Buffer", __LINE__, __FILE__, err);

	// Load skybox
	int skyboxWidth, skyboxHeight;
	loadSkyboxTexture(skyboxImages, &skyboxWidth, &skyboxHeight);

	// Config buffer
	imageConfig.skyboxSize = { skyboxWidth, skyboxHeight };
	imageConfigBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(imageConfig), &imageConfig, &err);

	// Skybox buffers
	size_t rgbsize = (size_t)skyboxWidth * skyboxHeight * 3;
	unsigned char* skbuf = new unsigned char[rgbsize * 6];
	for (int i = 0; i < 6; ++i) {
		memcpy(skbuf + (i * rgbsize), skyboxImages[i], rgbsize * sizeof(unsigned char));
	}
	skyboxBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 6 * rgbsize * sizeof(unsigned char), skbuf, &err);
	cl::printErrorMsg("Image Resolver Skybox Buffer", __LINE__, __FILE__, err);
	delete[] skbuf;

	outputImageBuffer = clCreateFromGLTexture(cl::context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &err);
	cl::printErrorMsg("Image Resolver Output Buffer", __LINE__, __FILE__, err);

	// Output image buffer
	if (texture == 0) {
		std::cout << "Texture is empty. Cannot create image resolver kernel without texture/output image buffer." << std::endl;
		return;
	}
	outputImageBuffer = clCreateFromGLTexture(cl::context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texture, &err);

	// Create results 2D array

	// Set kernel args
	err = clSetKernelArg(getKernel(), 0, sizeof(outputImageBuffer), &outputImageBuffer);
	cl::printErrorMsg("Output Buffer Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 1, sizeof(configBuffer), &configBuffer);
	cl::printErrorMsg("Config Buffer Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 2, sizeof(imageConfigBuffer), &imageConfigBuffer);
	cl::printErrorMsg("Image Config Buffer Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 3, sizeof(*world->getBufferPtr()), world->getBufferPtr());
	cl::printErrorMsg("World Buffer Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 4, sizeof(skyboxBuffer), &skyboxBuffer);
	cl::printErrorMsg("Skybox Buffer Kernel Arg", __LINE__, __FILE__, err);
}

cl_event RayTraceKernel::update() {
	cl_int err = clEnqueueWriteBuffer(cl::queue, configBuffer, true, 0, sizeof(RayConfig), config, 0, NULL, &updateEvent);
	cl::printErrorMsg("Ray Config Update", __LINE__, __FILE__, err);
	return updateEvent;
}

cl_event RayTraceKernel::queue(cl_uint num_events, cl_event* wait_events) {
	const size_t workgroupOffset[2] = { 0, 0 };
	const size_t workgroupSize[2] = { config->width, config->height };
	cl_int err = clEnqueueNDRangeKernel(cl::queue, getKernel(), 2, workgroupOffset, workgroupSize, NULL, num_events, wait_events, &queueEvent);
	return queueEvent;
}

void RayTraceKernel::destroy() {
}
