#pragma once
#include <CL/opencl.h>

__declspec (align(16)) struct Material {
	cl_float3 diffuse;
	cl_float specular;
	cl_float reflectivity;
	cl_float opacity;
	cl_float refractiveIndex;
};