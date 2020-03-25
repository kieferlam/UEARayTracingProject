#pragma once

#define NOMINMAX

#include <CL/opencl.h>
#include <vector>
#include <fstream>
#include <string>
#include "cl_helper.h"
#include "World.h"
#include "Mesh.h"

class World;
class Mesh;
struct ModelStruct;

class Model
{

	std::vector<Mesh> meshes;

	ModelStruct * modelStruct;

public:
	Model();
	~Model();

	void loadFromFile(const char* filename, World* world, float scale);

};

