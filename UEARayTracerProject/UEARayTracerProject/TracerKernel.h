#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include <time.h>
#include <stdio.h>
#include "cl_helper.h"

#define MAX_SPHERES (64)

struct SphereStruct {
	cl_float3 position;
	cl_float3 colour;
	cl_float radius;
	cl_float padding1;
	cl_float padding2;
	cl_float padding3;
};

struct KernelInputStruct {
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

	size_t worksize[2];

	KernelInputStruct kernelInputStruct;

public:

	bool create(GLuint texture, int width, int height);

	void writeKernelInput(cl_bool block);

	void trace(cl_bool block);

};

