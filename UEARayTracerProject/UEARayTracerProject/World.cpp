#include "World.h"
#include <stddef.h>

cl_mem _world_createBuffer(cl_mem_flags flags, size_t size, void * data, cl_int* err) {
	return clCreateBuffer(cl::context, flags, size > 0 ? size : 1, size > 0 ? data : NULL, err);
}

template<typename T>
void* _world_vectorFirstPtr(std::vector<T> & vector) {
	if (vector.size() > 0) return &vector[0];
	return NULL;
}

void World::create() {
	cl_int err;

	writeEvent = NULL;

	worldBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(WorldStruct), &world, &err);
	cl::printErrorMsg("Create World Buffer", __LINE__, __FILE__, err);

	vertexBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3) * vertices.size(), _world_vectorFirstPtr(vertices), &err);
	cl::printErrorMsg("Create Vertex Buffer", __LINE__, __FILE__, err);

	materialBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Material) * materials.size(), _world_vectorFirstPtr(materials), &err);
	cl::printErrorMsg("Create Material Buffer", __LINE__, __FILE__, err);

	sphereBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Sphere) * spheres.size(), _world_vectorFirstPtr(spheres), &err);
	cl::printErrorMsg("Create Sphere Buffer", __LINE__, __FILE__, err);

	triangleBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Triangle) * triangles.size(), _world_vectorFirstPtr(triangles), &err);
	cl::printErrorMsg("Create Triangle Buffer", __LINE__, __FILE__, err);

	modelBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(ModelStruct) * models.size(), _world_vectorFirstPtr(models), &err);
	cl::printErrorMsg("Create Model Buffer", __LINE__, __FILE__, err);

	triangleGridBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * triangleGrid.size(), _world_vectorFirstPtr(triangleGrid), &err);
	cl::printErrorMsg("Create Triangle Grid Buffer", __LINE__, __FILE__, err);

	triangleCountGridBuffer = _world_createBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * triangleCountGrid.size(), _world_vectorFirstPtr(triangleCountGrid), &err);
	cl::printErrorMsg("Create Triangle Count Grid Buffer", __LINE__, __FILE__, err);
}

ModelStruct* World::addModel(ModelStruct modelStruct)
{
	models.push_back(modelStruct);
	world.numModels = models.size();
	return &models.back();
}

unsigned int World::addSphere(cl_float3 position, cl_float radius, unsigned int material) {
	Sphere s = { position, radius, material };
	spheres.push_back(s);
	world.numSpheres = spheres.size();
	return spheres.size() - 1;
}

Sphere* World::getSphere(unsigned int index) {
	return &spheres[index];
}

unsigned int World::addTriangle(unsigned int i0, unsigned int i1, unsigned int i2) {
	// Get vertices
	cl_float3 v0 = vertices[i0];
	cl_float3 v1 = vertices[i1];
	cl_float3 v2 = vertices[i2];
	// Compute normal
	triangles.push_back({_world_computeTriangleNormal(v0, v1, v2), { i0, i1, i2 } });
	world.numTriangles = triangles.size();
	return triangles.size()-1;
}

unsigned int World::addTriangle(cl_uint3 face, cl_float3 normal)
{
	triangles.push_back({ normal, face });
	world.numTriangles = triangles.size();
	return triangles.size();
}

Triangle* World::getTriangle(unsigned int index)
{
	return &triangles[index];
}

unsigned int World::addVertex(cl_float3 vertex)
{
	vertices.push_back(vertex);
	return vertices.size()-1;
}

unsigned int World::addMaterial(Material m)
{
	materials.push_back(m);
	return materials.size() - 1;
}

void World::addTriangleGrid(unsigned int* gridOffset, unsigned int* countOffset) {
	*gridOffset = triangleGrid.size();
	*countOffset = triangleCountGrid.size();
	triangleGrid.resize(triangleGrid.size() + GRID_CELL_COUNT * GRID_MAX_TRIANGLES_PER_CELL, 0);
	triangleCountGrid.resize(triangleCountGrid.size() + GRID_CELL_COUNT, 0);
}

void World::addTriangleToGrid(unsigned int triangle, unsigned int offset) {
	triangleGrid[offset*GRID_MAX_TRIANGLES_PER_CELL + triangleCountGrid[offset]] = triangle;
	triangleCountGrid[offset]++;
}

void World::setTriangleMaterial(unsigned int triangle, unsigned int material)
{
	triangles[triangle].materialIndex = material;
}

cl_event World::update() {
	cl_int err = clEnqueueWriteBuffer(cl::queue, worldBuffer, false, 0, sizeof(world), &world, 0, NULL, &writeEvent);
	cl::printErrorMsg("Update World Buffer", __LINE__, __FILE__, err);
	return writeEvent;
}

cl_event World::updateSpheres(unsigned int sphereStartIndex, unsigned int numSpheres) {
	cl_int err = clEnqueueWriteBuffer(cl::queue, sphereBuffer, false, sizeof(Sphere) * sphereStartIndex, sizeof(Sphere) * numSpheres, &spheres[sphereStartIndex], 0, NULL, &writeEvent);
	cl::printErrorMsg("Update Spheres [" + std::to_string(sphereStartIndex) + ", " + std::to_string(numSpheres) + "]", __LINE__, __FILE__, err);
	return writeEvent;
}
