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

#define GRID_CELL_ROW_COUNT (16)
#define GRID_MAX_TRIANGLES_PER_CELL (8)

#define SQ(x) ((x)*(x)) 
#define CUBE(x) ((x)*(x)*(x))

inline cl_float _world_computeLength(cl_float3 vector) {
	return sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

inline cl_float3 _world_normalise(cl_float3 vector) {
	cl_float length = _world_computeLength(vector);
	return { vector.x / length, vector.y / length, vector.z / length };
}

inline cl_float3 _world_cross(cl_float3 a, cl_float3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

inline cl_float3 operator-(const cl_float3& v) {
	return { -v.x, -v.y, -v.z };
}

inline cl_float3 operator-(const cl_float3& a, const cl_float3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z};
}

inline cl_float3 _world_computeTriangleNormal(cl_float3 v0, cl_float3 v1, cl_float3 v2) {
	cl_float3 v0v1 = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
	cl_float3 v0v2 = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };
	return _world_normalise(_world_cross(v0v1, v0v2));
}


class Model;

__declspec (align(16)) struct ModelStruct {
	cl_uint triangleGrid[CUBE(GRID_CELL_ROW_COUNT) * GRID_MAX_TRIANGLES_PER_CELL]; // Stores index of triangles in each grid cell
	cl_uchar cellTriangleCount[CUBE(GRID_CELL_ROW_COUNT)]; // Stores the count of triangles in each grid cell
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

