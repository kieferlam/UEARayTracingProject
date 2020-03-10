#include "World.h"
#include <stddef.h>

void World::create() {
	cl_int err;

	writeEvent = NULL;

	worldBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(WorldStruct), &world, &err);
	cl::printErrorMsg("Create World Buffer", __LINE__, __FILE__, err);
}

int World::addSphere(cl_float3 position, cl_float radius, Material material) {
	world.spheres[world.numSpheres].position = position;
	world.spheres[world.numSpheres].radius = radius;
	world.spheres[world.numSpheres++].material = material;
	return world.numSpheres - 1;
}

Sphere* World::getSphere(int index) {
	return world.spheres + index;
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
