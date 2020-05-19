#pragma once
#include <CL/opencl.h>
#include <vector>
#include "cl_helper.h"
#include "Sphere.h"
#include "Material.h"
#include "Triangle.h"
#include "Model.h"

#define SQ(x) ((x)*(x)) 
#define CUBE(x) ((x)*(x)*(x))

#define GRID_CELL_DEPTH (1)
#define GRID_MAX_TRIANGLES_PER_CELL (64) // MAKE SURE THIS IS A MULTIPLE OF 16

inline constexpr int static_pow(const int base, const int exp) { return (exp == 0) ? 1 : base * static_pow(base, exp-1); }
inline constexpr int static_numrays(const int numchildren, const int bounce) { return (1 - static_pow(numchildren, bounce + 1)) / (1-numchildren); }

constexpr int GRID_CELL_ROW_COUNT = static_pow(2, GRID_CELL_DEPTH);
constexpr int GRID_CELL_COUNT = CUBE(GRID_CELL_ROW_COUNT);

inline constexpr unsigned int getGridOffset(const cl_int3 coord) { return coord.x * SQ(GRID_CELL_ROW_COUNT) + coord.y * GRID_CELL_ROW_COUNT + coord.z; }

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
	cl_float2 bounds[8]; // Only 7 axis but the 8th is for padding (struct alignment)
	cl_uint triangleOffset;
	cl_uint pad1[3];
	cl_uint numTriangles;
	cl_uint pad2[3];
	cl_uint triangleGridOffset;
	cl_uint pad3[3];
	cl_uint triangleCountOffset;
	cl_uint pad4[3];
};

__declspec (align(16)) struct WorldStruct {
	cl_uint numRays;
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

	std::vector<Triangle> triangles;
	cl_mem triangleBuffer;

	std::vector<Sphere> spheres;
	cl_mem sphereBuffer;

	std::vector<ModelStruct> models;
	cl_mem modelBuffer;

	std::vector<unsigned int> triangleGrid;
	cl_mem triangleGridBuffer;

	std::vector<unsigned int> triangleCountGrid;
	cl_mem triangleCountGridBuffer;

public:

	void create();

	inline cl_mem* getBufferPtr() { return &worldBuffer; }

	inline cl_mem* getVertexBufferPtr() { return &vertexBuffer; }

	inline cl_mem* getMaterialBufferPtr() { return &materialBuffer; }

	inline cl_mem* getSphereBufferPtr() { return &sphereBuffer; }
	inline cl_mem* getTriangleBufferPtr() { return &triangleBuffer; }
	inline cl_mem* getModelBufferPtr() { return &modelBuffer; }

	inline cl_mem* getTriangleGridPtr() { return &triangleGridBuffer; }

	inline cl_mem* getTriangleCountPtr() { return &triangleCountGridBuffer; }

	inline std::vector<cl_float3>& getVertexBuffer() { return vertices; }

	inline std::vector<Material>& getMaterialBuffer() { return materials; }

	inline std::vector<Sphere>& getSpheres() { return spheres; }

	inline cl_uint getTriangleCount() { return world.numTriangles; }

	ModelStruct* addModel(ModelStruct modelStruct);

	unsigned int addSphere(cl_float3 position, cl_float radius, unsigned int material);

	Sphere* getSphere(unsigned int index);

	unsigned int addTriangle(unsigned int i0, unsigned int i1, unsigned int i2);

	unsigned int addTriangle(cl_uint3 face, cl_float3 normal);

	Triangle* getTriangle(unsigned int index);

	unsigned int addVertex(cl_float3 vertex);

	unsigned int addMaterial(Material m);

	void addTriangleGrid(unsigned int * gridOffset, unsigned int * countOffset);

	void addTriangleToGrid(unsigned int triangle, unsigned int offset);

	inline std::vector<unsigned int>& getTriangleGrid() { return triangleGrid; }

	inline std::vector<unsigned int>& getTriangleCountGrid() { return triangleCountGrid; }

	void setTriangleMaterial(unsigned int triangle, unsigned int material);

	cl_event update();

	cl_event updateSpheres(unsigned int sphereStartIndex, unsigned int numSpheres);

};

