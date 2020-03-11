#pragma once
#include <CL/opencl.h>
#include <vector>
#include "cl_helper.h"
#include "Sphere.h"
#include "Material.h"
#include "Triangle.h"

#define MAX_SPHERES (8)
#define MAX_TRIANGLES (65536)

__declspec (align(16)) struct WorldStruct {
	Sphere spheres[MAX_SPHERES];
	Triangle triangles[MAX_TRIANGLES];
	int numSpheres;
	int numTriangles;
};

class World {

	WorldStruct world;

	cl_mem worldBuffer;

	cl_event writeEvent;

	std::vector<cl_float3> vertices;
	cl_mem vertexBuffer;

public:

	void create();

	inline cl_mem* getBufferPtr() { return &worldBuffer; }

	inline cl_mem* getVertexBufferPtr() { return &vertexBuffer; }

	int addSphere(cl_float3 position, cl_float radius, Material material);

	Sphere* getSphere(int index);

	cl_event update();

	cl_event updateSpheres(int sphereStartIndex, int numSpheres);

};

