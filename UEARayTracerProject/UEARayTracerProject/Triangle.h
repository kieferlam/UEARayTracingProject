#pragma once
#include <CL/opencl.h>
#include "Material.h"

__declspec (align(16)) struct Triangle {
	cl_float3 normal;
	cl_int3 face;
	cl_int materialIndex;
};