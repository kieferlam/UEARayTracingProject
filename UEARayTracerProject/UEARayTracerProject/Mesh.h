#pragma once
#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <string>
#include "cl_helper.h"
#include "World.h"

class Mesh
{

	std::vector<unsigned int> triangles;

public:
	Mesh();
	~Mesh();

	std::string name;
	
	inline void addTriangle(unsigned int triangle) { triangles.push_back(triangle); }

};

