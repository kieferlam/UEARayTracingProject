#include "World.h"
#include <stddef.h>

void World::create() {
	cl_int err;

	writeEvent = NULL;

	worldBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(WorldStruct), &world, &err);
	cl::printErrorMsg("Create World Buffer", __LINE__, __FILE__, err);

	vertexBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3) * vertices.size(), &vertices[0], &err);
	cl::printErrorMsg("Create Vertex Buffer", __LINE__, __FILE__, err);

	materialBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(Material) * materials.size(), &materials[0], &err);
	cl::printErrorMsg("Create Material Buffer", __LINE__, __FILE__, err);
}

ModelStruct* World::addModel(ModelStruct modelStruct)
{
	world.models[world.numModels++] = modelStruct;
	return world.models + (world.numModels - 1);
}

unsigned int World::addSphere(cl_float3 position, cl_float radius, unsigned int material) {
	world.spheres[world.numSpheres].position = position;
	world.spheres[world.numSpheres].radius = radius;
	world.spheres[world.numSpheres++].material = material;
	return world.numSpheres - 1;
}

Sphere* World::getSphere(unsigned int index) {
	return world.spheres + index;
}

unsigned int World::addTriangle(unsigned int i0, unsigned int i1, unsigned int i2)
{
	// Get vertices
	cl_float3 v0 = vertices[i0];
	cl_float3 v1 = vertices[i1];
	cl_float3 v2 = vertices[i2];
	// Compute normal
	world.triangles[world.numTriangles++] = { _world_computeTriangleNormal(v0, v1, v2), {i0, i1, i2} };
	return world.numTriangles-1;
}

unsigned int World::addTriangle(cl_uint3 face, cl_float3 normal)
{
	world.triangles[world.numTriangles++] = { normal, face };
	return world.numTriangles-1;
}

Triangle* World::getTriangle(unsigned int index)
{
	return world.triangles + index;
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

void World::setTriangleMaterial(unsigned int triangle, unsigned int material)
{
	world.triangles[triangle].materialIndex = material;
}

cl_event World::update() {
	cl_int err = clEnqueueWriteBuffer(cl::queue, worldBuffer, false, 0, sizeof(world), &world, 0, NULL, &writeEvent);
	cl::printErrorMsg("Update World Buffer", __LINE__, __FILE__, err);
	return writeEvent;
}

cl_event World::updateSpheres(unsigned int sphereStartIndex, unsigned int numSpheres) {
	cl_int err = clEnqueueWriteBuffer(cl::queue, worldBuffer, false, offsetof(struct WorldStruct, spheres) + sizeof(Sphere) * sphereStartIndex, sizeof(Sphere) * numSpheres, &world.spheres + sphereStartIndex, 0, NULL, &writeEvent);
	cl::printErrorMsg("Update Spheres [" + std::to_string(sphereStartIndex) + ", " + std::to_string(numSpheres) + "]", __LINE__, __FILE__, err);
	return writeEvent;
}
