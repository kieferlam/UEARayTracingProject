#pragma once
#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <string>
#include "cl_helper.h"
#include "World.h"
#include "Mesh.h"

class Model
{

	std::vector<Mesh> meshes;

public:
	Model();
	~Model();

	void loadFromFile(const char* filename, World* world, float scale);

};
