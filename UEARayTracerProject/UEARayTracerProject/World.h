#pragma once
#include <CL/opencl.h>
#include "cl_helper.h"
#include "Sphere.h"
#include "Material.h"

#define MAX_SPHERES (8)

__declspec (align(16)) struct WorldStruct {
	Sphere spheres[MAX_SPHERES];
	int numSpheres;
};

class World {

	WorldStruct world;

	cl_mem worldBuffer;

	cl_event writeEvent;

public:

	void create();

	inline cl_mem* getBufferPtr() { return &worldBuffer; }

	int addSphere(cl_float3 position, cl_float radius, Material material);

	Sphere* getSphere(int index);

	cl_event update();

	cl_event updateSpheres(int sphereStartIndex, int numSpheres);

};

