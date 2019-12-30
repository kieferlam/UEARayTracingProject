#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include <time.h>
#include <stdio.h>
#include "cl_helper.h"

#define MAX_SPHERES (8)

__declspec (align(16)) struct Material {
	cl_float3 diffuse;
	cl_float reflectivity;
	cl_float opacity;
	cl_float refractiveIndex;
};

__declspec (align(16)) struct SphereStruct {
	Material material;
	cl_float3 position;
	cl_float radius;
};

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
	SphereStruct spheres[MAX_SPHERES];
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

};

