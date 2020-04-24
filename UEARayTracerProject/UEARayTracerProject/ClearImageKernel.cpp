#include "ClearImageKernel.h"

ClearImageKernel::ClearImageKernel() : CLKernel("ClearImage") {
}

ClearImageKernel::~ClearImageKernel() {
}

void ClearImageKernel::create() {
	cl_int err;

	err = clSetKernelArg(getKernel(), 0, sizeof(*imageBuffer), imageBuffer);
	cl::printErrorMsg("Clear Image Arg", __LINE__, __FILE__, err);

	err = clSetKernelArg(getKernel(), 1, sizeof(*imageConfigBuffer), imageConfigBuffer);
	cl::printErrorMsg("Clear Image Config Arg", __LINE__, __FILE__, err);
}

cl_event ClearImageKernel::update() {
	return cl_event();
}

cl_event ClearImageKernel::queue(cl_uint num_events, cl_event* wait_events) {
	cl_event queueEvent;

	const size_t workgroupOffset[2] = { 0, 0 };
	const size_t workgroupSize[2] = { cfg->res.x, cfg->res.y };
	cl_int err = clEnqueueNDRangeKernel(cl::queue, getKernel(), 2, workgroupOffset, workgroupSize, NULL, num_events, wait_events, &queueEvent);
	cl::printErrorMsg("Clear Image Kernel Queue", __LINE__, __FILE__, err);
	return queueEvent;
}

void ClearImageKernel::destroy() {
}
