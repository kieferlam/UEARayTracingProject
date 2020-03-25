#pragma once
#include <CL/opencl.h>
#include <vector>
#include "cl_helper.h"
#include "Sphere.h"
#include "Material.h"
#include "Triangle.h"
#include "Model.h"

#define MAX_SPHERES (8)
#define MAX_TRIANGLES (65536)
#define MAX_MODELS (32)

class Model;

__declspec (align(16)) struct ModelStruct {
	cl_float2 bounds[7];
	cl_uint triangleOffset;
	cl_uint numTriangles;
};

__declspec (align(16)) struct WorldStruct {
	Sphere spheres[MAX_SPHERES];
	Triangle triangles[MAX_TRIANGLES];
	ModelStruct models[MAX_MODELS];
	cl_uint numSpheres;
	cl_uint numTriangles;
	cl_uint numModels;
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

	inline cl_uint getTriangleCount() { return world.numTriangles; }

	ModelStruct* addModel(ModelStruct modelStruct);

	unsigned int addSphere(cl_float3 position, cl_float radius, unsigned int material);

	Sphere* getSphere(unsigned int index);

	unsigned int addTriangle(unsigned int i0, unsigned int i1, unsigned int i2);

	unsigned int addTriangle(cl_uint3 face, cl_float3 normal);

	Triangle* getTriangle(unsigned int index);

	unsigned int addVertex(cl_float3 vertex);

	unsigned int addMaterial(Material m);

	void setTriangleMaterial(unsigned int triangle, unsigned int material);

	cl_event update();

	cl_event updateSpheres(unsigned int sphereStartIndex, unsigned int numSpheres);

};

