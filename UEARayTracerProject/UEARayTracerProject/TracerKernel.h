#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include "cl_helper.h"

class TracerKernel {

	const std::string kernelName = "TracerMain";

	cl_kernel kernel;

	cl_mem paramInput;

	cl_mem outputImageBuffer;

	struct {} kernelInputStruct;

public:

	bool create(GLuint texture);

};

