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
class Model;

class Mesh
{

	std::vector<unsigned int> triangles;

	cl_float2 bounds[7];

public:
	Mesh();
	~Mesh();

	std::string name;
	
	inline void addTriangle(unsigned int triangle) { triangles.push_back(triangle); }

	void createBoundingVolume(const Triangle * triangles, const std::vector<cl_float3>& vertices);

	inline cl_float2* getBounds() { return bounds; }

	inline cl_float2 getBounds(int index) { return bounds[index]; }

	void getTrianglesInGridCell(cl_float2* bounds, int cellIndex, cl_uint* cellTriangles, cl_uchar* triangleCount);

};

