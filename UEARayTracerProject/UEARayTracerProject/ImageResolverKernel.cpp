#include "ImageResolverKernel.h"
#include <SOIL.h>

ImageResolverKernel::ImageResolverKernel() : CLKernel("ResolveImage") {
}

ImageResolverKernel::~ImageResolverKernel() {
}

void _image_loadSkyboxTexture(unsigned char** skyboxImages, int* width, int* height) {

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

void ImageResolverKernel::create() {
	cl_int err;

	// Load skybox
	int skyboxWidth, skyboxHeight;
	_image_loadSkyboxTexture(skyboxImages, &skyboxWidth, &skyboxHeight);

	// Config buffer
	config.skyboxSize = { skyboxWidth, skyboxHeight };
	configBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(config), &config, &err);

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

	// Set Kernel Args

	err = clSetKernelArg(getKernel(), 0, sizeof(outputImageBuffer), &outputImageBuffer);
	cl::printErrorMsg("Image Resolver Output Image Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 1, sizeof(*rayConfig), rayConfig);
	cl::printErrorMsg("Image Resolver Ray Config Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 2, sizeof(configBuffer), &configBuffer);
	cl::printErrorMsg("Image Resolver Config Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 3, sizeof(*rayBuffer), rayBuffer);
	cl::printErrorMsg("Image Resolver Ray Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 4, sizeof(skyboxBuffer), &skyboxBuffer);
	cl::printErrorMsg("Image Resolver Skybox Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 5, sizeof(*materialBuffer), materialBuffer);
	cl::printErrorMsg("Image Resolver Material Buffer Arg", __LINE__, __FILE__, err);
}

cl_event ImageResolverKernel::update() {
	return NULL;
}

void ImageResolverKernel::destroy() {
}

cl_event ImageResolverKernel::queue(cl_uint num_events, cl_event* wait_events) {
	const size_t workgroupOffset[2] = { 0, 0 };
	const size_t workgroupSize[2] = { config.res.x, config.res.y};
	cl_int err = clEnqueueNDRangeKernel(cl::queue, getKernel(), 2, workgroupOffset, workgroupSize, NULL, num_events, wait_events, &queueEvent);
	cl::printErrorMsg("Image Resolver Kernel Queue", __LINE__, __FILE__, err);
	return queueEvent;
}
