#include "ResetKernel.h"

ResetKernel::ResetKernel() : CLKernel("ResetRays")
{
}

ResetKernel::~ResetKernel()
{
}

void ResetKernel::create()
{
	cl_int err;

	err = clSetKernelArg(getKernel(), 0, sizeof(*configBuffer), configBuffer);
	cl::printErrorMsg("Reset Kernel Config Kernel Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 1, sizeof(*resultBuffer), resultBuffer);
	cl::printErrorMsg("Reset Kernel Result Kernel Arg", __LINE__, __FILE__, err);

}

cl_event ResetKernel::update()
{
	return NULL;
}

cl_event ResetKernel::queue(cl_uint num_events, cl_event* wait_events)
{
	const unsigned int numrays = static_numrays(NUM_RAY_CHILDREN, config->bounces);
	const size_t workgroupOffset[1] = { 0};
	const size_t workgroupSize[1] = { config->width * config->height * numrays };
	cl_int err = clEnqueueNDRangeKernel(cl::queue, getKernel(), 1, workgroupOffset, workgroupSize, NULL, num_events, wait_events, &queueEvent);
	cl::printErrorMsg("Enqueue Reset Kernel", __LINE__, __FILE__, err);
	return queueEvent;
}

void ResetKernel::destroy()
{
}
