#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include "cl_helper.h"

#define MAX_SPHERES (64)

struct SphereStruct {
	cl_float3 position;
	cl_float4 colour;
	cl_float radius;
};

class TracerKernel {

	const std::string kernelName = "TracerMain";

	cl_kernel kernel;

	cl_mem paramInput;

	cl_mem outputImageBuffer;

	size_t worksize[2];

	struct {
		SphereStruct spheres[MAX_SPHERES];
		cl_int numSpheres;
	} kernelInputStruct;

public:

	bool create(GLuint texture, int width, int height);

	void trace(bool block);

};

