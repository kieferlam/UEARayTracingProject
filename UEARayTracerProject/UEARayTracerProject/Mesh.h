#pragma once

#define NOMINMAX

#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <limits>
#include "cl_helper.h"
#include "World.h"

const float SQRT33 = sqrt(3.0f) / 3.0f;

const cl_float3 BVH_PlaneNormals[] = {
	{1.0f, 0.0f, 0.0f},
	{0.0f, 1.0f, 0.0f},
	{0.0f, 0.0f, 1.0f},
	{SQRT33, SQRT33, SQRT33},
	{-SQRT33, SQRT33, SQRT33},
	{-SQRT33, -SQRT33, SQRT33},
	{SQRT33, -SQRT33, SQRT33},
};

class World;

struct OctreeCell {
	std::vector<unsigned int> triangles;
	OctreeCell* children[8] = { nullptr };
	cl_float2 bounds[3];
	unsigned int depth;
};

class Mesh
{

	cl_float2 bounds[7] = { 0 };

	OctreeCell octree;

	std::vector<OctreeCell*> leafCells;

public:
	Mesh();
	~Mesh();

	std::string name;
	
	inline void addTriangle(unsigned int triangle) { octree.triangles.push_back(triangle); }

	void createBoundingVolume(const Triangle * triangles, const std::vector<cl_float3>& vertices);

	inline cl_float2* getBounds() { return bounds; }

	inline cl_float2 getBounds(int index) { return bounds[index]; }

	inline size_t getTriangleCount() { return octree.triangles.size(); }

	void getTrianglesInGridCell(World * world, cl_float2* bounds, cl_uint* cellTriangles, cl_uchar* triangleCount);

	void constructOctree(World* world, int depth, const cl_float2* bounds);

	inline const std::vector<OctreeCell*>& getLeafNodes() { return leafCells; }

};

