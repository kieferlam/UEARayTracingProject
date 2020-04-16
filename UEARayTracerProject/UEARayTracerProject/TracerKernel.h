#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include <time.h>
#include <stdio.h>
#include "cl_helper.h"
#include "Sphere.h"
#include "Material.h"

__declspec (align(32)) struct RTConfig {
	cl_int2 skyboxSize;
	cl_int bounceLimit = 3;
	cl_int skybox = true;
	cl_int shadows = true;
	cl_int reflection = true;
	cl_int refraction = true;
};

__declspec (align(16)) struct KernelInputStruct {
	cl_float aspect;
	cl_float width;
	cl_float height;
	cl_float screenDistance;
	cl_float3 camera;
	Sphere spheres[8];
	int numSpheres;
};

class TracerKernel {

	const std::string kernelName = "TracerMain";

	cl_kernel kernel;

	cl_mem inputKernelBuffer;

	cl_mem outputImageBuffer;

	cl_mem configBuffer;

	unsigned char* skyboxImages[6] = { nullptr ,nullptr ,nullptr ,nullptr ,nullptr ,nullptr };
	cl_mem skyboxBuffer;

	size_t worksize[2];

	KernelInputStruct kernelInputStruct;

	RTConfig config;

public:

	bool create(GLuint texture, int width, int height);

	void writeKernelInput(cl_bool block);

	void trace(cl_bool block);

	void keyevent(int key, int scancode, int action, int mods);

};

