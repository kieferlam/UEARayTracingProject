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

int World::addSphere(cl_float3 position, cl_float radius, int material) {
	world.spheres[world.numSpheres].position = position;
	world.spheres[world.numSpheres].radius = radius;
	world.spheres[world.numSpheres++].material = material;
	return world.numSpheres - 1;
}

Sphere* World::getSphere(int index) {
	return world.spheres + index;
}

cl_float _world_computeLength(cl_float3 vector) {
	return sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);
}

cl_float3 _world_normalise(cl_float3 vector) {
	cl_float length = _world_computeLength(vector);
	return { vector.x / length, vector.y / length, vector.z / length };
}

cl_float3 _world_cross(cl_float3 a, cl_float3 b) {
	return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
}

cl_float3 _world_computeTriangleNormal(cl_float3 v0, cl_float3 v1, cl_float3 v2) {
	cl_float3 v0v1 = { v1.x - v0.x, v1.y - v0.y, v1.z - v0.z };
	cl_float3 v0v2 = { v2.x - v0.x, v2.y - v0.y, v2.z - v0.z };
	return _world_normalise(_world_cross(v0v1, v0v2));
}

int World::addTriangle(int i0, int i1, int i2)
{
	// Get vertices
	cl_float3 v0 = vertices[i0];
	cl_float3 v1 = vertices[i1];
	cl_float3 v2 = vertices[i2];
	// Compute normal
	world.triangles[world.numTriangles++] = { _world_computeTriangleNormal(v0, v1, v2), {i0, i1, i2} };
	return world.numTriangles-1;
}

Triangle* World::getTriangle(int index)
{
	return world.triangles + index;
}

int World::addVertex(cl_float3 vertex)
{
	vertices.push_back(vertex);
	return vertices.size()-1;
}

int World::addMaterial(Material m)
{
	materials.push_back(m);
	return materials.size() - 1;
}

void World::setTriangleMaterial(int triangle, int material)
{
	world.triangles[triangle].materialIndex = material;
}

cl_event World::update() {
	cl_int err = clEnqueueWriteBuffer(cl::queue, worldBuffer, false, 0, sizeof(world), &world, 0, NULL, &writeEvent);
	cl::printErrorMsg("Update World Buffer", __LINE__, __FILE__, err);
	return writeEvent;
}

cl_event World::updateSpheres(int sphereStartIndex, int numSpheres) {
	cl_int err = clEnqueueWriteBuffer(cl::queue, worldBuffer, false, offsetof(struct WorldStruct, spheres) + sizeof(Sphere) * sphereStartIndex, sizeof(Sphere) * numSpheres, &world.spheres + sphereStartIndex, 0, NULL, &writeEvent);
	cl::printErrorMsg("Update Spheres [" + std::to_string(sphereStartIndex) + ", " + std::to_string(numSpheres) + "]", __LINE__, __FILE__, err);
	return writeEvent;
}
