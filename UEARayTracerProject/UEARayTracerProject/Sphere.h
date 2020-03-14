#pragma once
#include <CL/opencl.h>
#include "Material.h"

__declspec (align(16)) struct Sphere {
	cl_float3 position;
	cl_float radius;
	int material;
};