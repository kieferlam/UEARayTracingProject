#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include "CLKernel.h"
#include "cl_helper.h"
#include "RARKernel.h"

#pragma once
class ResetKernel : public CLKernel
{
	RayConfig* config;
	cl_mem* resultBuffer;
	cl_mem* configBuffer;

	cl_event queueEvent;

public:
	ResetKernel();
	~ResetKernel();

	inline void setRayBuffer(cl_mem* ptr) { resultBuffer = ptr; }
	inline void setConfigBuffer(cl_mem* ptr) { configBuffer = ptr; }
	inline void setConfig(RayConfig* ptr) { config = ptr; }

	// Inherited via CLKernel
	virtual void create() override;
	virtual cl_event update() override;
	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;
	virtual void destroy() override;
};

