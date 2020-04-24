#include "TestKernel.h"

TestKernel::TestKernel() : CLKernel("TestStructs")
{
}

TestKernel::~TestKernel()
{
}

void TestKernel::create()
{
	cl_int err;

	in_modelBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(ModelStruct), &in_model, &err);
	cl::printErrorMsg("Test Kernel Input Model Buffer", __LINE__, __FILE__, err);
	out_modelBuffer = clCreateBuffer(cl::context, CL_MEM_READ_WRITE, sizeof(ModelStruct), NULL, &err);
	cl::printErrorMsg("Test Kernel Output Model Buffer", __LINE__, __FILE__, err);

	in_worldBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(WorldStruct), &in_world, &err);
	cl::printErrorMsg("Test Kernel Input World Buffer", __LINE__, __FILE__, err);
	out_worldBuffer = clCreateBuffer(cl::context, CL_MEM_READ_WRITE, sizeof(WorldStruct), NULL, &err);
	cl::printErrorMsg("Test Kernel Output World Buffer", __LINE__, __FILE__, err);

	clSetKernelArg(getKernel(), 0, sizeof(in_modelBuffer), &in_modelBuffer);
	clSetKernelArg(getKernel(), 1, sizeof(out_modelBuffer), &out_modelBuffer);
	clSetKernelArg(getKernel(), 2, sizeof(in_worldBuffer), &in_worldBuffer);
	clSetKernelArg(getKernel(), 3, sizeof(out_worldBuffer), &out_worldBuffer);
}

cl_event TestKernel::update()
{
	cl_event writeevent;
	cl_int err = clEnqueueWriteBuffer(cl::queue, in_modelBuffer, true, 0, sizeof(ModelStruct), &in_model, 0, NULL, &writeevent);
	cl::printErrorMsg("Test Kernel Update Model", __LINE__, __FILE__, err);
	err = clEnqueueWriteBuffer(cl::queue, in_worldBuffer, true, 0, sizeof(WorldStruct), &in_world, 0, NULL, &writeevent);
	cl::printErrorMsg("Test Kernel Update World", __LINE__, __FILE__, err);
	return writeevent;
}

cl_event TestKernel::queue(cl_uint num_events, cl_event* wait_events)
{
	const size_t workgroupOffset = 0;
	const size_t workgroupSize = 1;
	cl_int err = clEnqueueNDRangeKernel(cl::queue, getKernel(), 1, &workgroupOffset, &workgroupSize, NULL, num_events, wait_events, &mainQueueEvent);
	cl::printErrorMsg("Test Kernel Queue", __LINE__, __FILE__, err);

	err = clEnqueueReadBuffer(cl::queue, out_modelBuffer, true, 0, sizeof(ModelStruct), &out_model, 0, NULL, NULL);
	cl::printErrorMsg("Test Kernel Read Model", __LINE__, __FILE__, err);

	err = clEnqueueReadBuffer(cl::queue, out_worldBuffer, true, 0, sizeof(WorldStruct), &out_world, 0, NULL, NULL);
	cl::printErrorMsg("Test Kernel Read World", __LINE__, __FILE__, err);

	// Do checks
	if (in_model.triangleGridOffset != out_model.triangleGridOffset) {
		log << "Mismatch in triangle grid offset \tExpected\t" << in_model.triangleGridOffset << "\tgot\t" << out_model.triangleGridOffset << std::endl;
	}

	if (in_model.triangleCountOffset != out_model.triangleCountOffset) {
		log << "Mismatch in triangle count offset \tExpected\t" << in_model.triangleCountOffset << "\tgot\t" << out_model.triangleCountOffset << std::endl;
	}

	for (int i = 0; i < 7; ++i) {
		if (in_model.bounds[i].x != out_model.bounds[i].x || in_model.bounds[i].y != out_model.bounds[i].y) {
			log << "Mismatch in triangle bounds i: " << i << "\tExpected\tx: " << in_model.bounds[i].x << " y: " << in_model.bounds[i].y << "\tgot\tx: " << out_model.bounds[i].x << " y: " << out_model.bounds[i].y << std::endl;
		}
	}

	if (in_model.triangleOffset != out_model.triangleOffset) {
		log << "Mismatch in triangle offset \tExpected\t" << in_model.triangleOffset << "\tgot\t" << out_model.triangleOffset << std::endl;
	}

	if (in_model.numTriangles != out_model.numTriangles) {
		log << "Mismatch in triangle count \tExpected\t" << in_model.numTriangles << "\tgot\t" << out_model.numTriangles << std::endl;
	}

	// World struct

	if (in_world.numSpheres != out_world.numSpheres) {
		log << "Mismatch in world numSpheres \tExpected\t" << in_world.numSpheres << "\tgot\t" << out_world.numSpheres << std::endl;
	}

	if (in_world.numTriangles != out_world.numTriangles) {
		log << "Mismatch in world numTriangles \tExpected\t" << in_world.numTriangles << "\tgot\t" << out_world.numTriangles << std::endl;
	}

	if (in_world.numModels != out_world.numModels) {
		log << "Mismatch in world numModels \tExpected\t" << in_world.numModels << "\tgot\t" << out_world.numModels << std::endl;
	}

	return mainQueueEvent;
}

void TestKernel::destroy()
{
}
