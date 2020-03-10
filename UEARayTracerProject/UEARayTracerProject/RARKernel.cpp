#include "RARKernel.h"
#include <math.h>

RARKernel::RARKernel() : CLKernel("RARTrace") {
}

RARKernel::~RARKernel() {
}

void RARKernel::read() {
	cl_int err = clEnqueueReadBuffer(cl::queue, outputBuffer, true, 0, sizeof(TraceResult) * config->width * config->height, results, 0, NULL, NULL);
	cl::printErrorMsg("Output Read Buffer", __LINE__, __FILE__, err);
}

void RARKernel::create() {

	int numrays = pow(2, config->bounces) - 1;

	// Assume kernel object has been created

	cl_int err;

	configBuffer = clCreateBuffer(cl::context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(*config), config, &err);
	cl::printErrorMsg("Config Buffer", __LINE__, __FILE__, err);

	// Create results 2D array

	outputBuffer = clCreateBuffer(cl::context, CL_MEM_READ_WRITE, sizeof(TraceResult) * config->width * config->height * numrays, NULL, &err);
	cl::printErrorMsg("Output Buffer", __LINE__, __FILE__, err);

	// Set kernel args
	err = clSetKernelArg(getKernel(), 0, sizeof(configBuffer), &configBuffer);
	cl::printErrorMsg("Config Buffer Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 1, sizeof(*world->getBufferPtr()), world->getBufferPtr());
	cl::printErrorMsg("World Buffer Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 2, sizeof(outputBuffer), &outputBuffer);
	cl::printErrorMsg("Output Buffer Kernel Arg", __LINE__, __FILE__, err);
}

cl_event RARKernel::update() {
	cl_int err = clEnqueueWriteBuffer(cl::queue, configBuffer, true, 0, sizeof(RayConfig), config, 0, NULL, &updateEvent);
	cl::printErrorMsg("Write Config Buffer", __LINE__, __FILE__, err);
	return updateEvent;
}

cl_event RARKernel::queue(cl_uint num_events, cl_event* wait_events) {
	const size_t workgroupOffset[2] = { 0, 0 };
	const size_t workgroupSize[2] = { config->width, config->height };
	cl_int err = clEnqueueNDRangeKernel(cl::queue, getKernel(), 2, workgroupOffset, workgroupSize, NULL, num_events, wait_events, &queueEvent);
	cl::printErrorMsg("Enqueue Primary Ray Kernel", __LINE__, __FILE__, err);
	return queueEvent;
}

void RARKernel::destroy() {

}
