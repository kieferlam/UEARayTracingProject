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

	std::vector<Material> materials;
	cl_mem materialBuffer;

public:

	void create();

	inline cl_mem* getBufferPtr() { return &worldBuffer; }

	inline cl_mem* getVertexBufferPtr() { return &vertexBuffer; }

	inline cl_mem* getMaterialBufferPtr() { return &materialBuffer; }

	inline std::vector<cl_float3>& getVertexBuffer() { return vertices; }

	inline std::vector<Material>& getMaterialBuffer() { return materials; }

	int addSphere(cl_float3 position, cl_float radius, int material);

	Sphere* getSphere(int index);

	int addTriangle(int i0, int i1, int i2);

	Triangle* getTriangle(int index);

	int addVertex(cl_float3 vertex);

	int addMaterial(Material m);

	void setTriangleMaterial(int triangle, int material);

	cl_event update();

	cl_event updateSpheres(int sphereStartIndex, int numSpheres);

};

