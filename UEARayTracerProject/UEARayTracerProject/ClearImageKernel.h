#pragma once
#include "CLKernel.h"
#include "ImageResolverKernel.h"

class ClearImageKernel :
	public CLKernel {

	cl_mem* imageConfigBuffer;
	cl_mem* imageBuffer;

	ImageConfig* cfg;

public:
	ClearImageKernel();
	~ClearImageKernel();

	inline void setImageConfig(ImageConfig* ptr) { cfg = ptr; }
	inline void setImageConfigBuffer(cl_mem* ptr) { imageConfigBuffer = ptr; }
	inline void setImage(cl_mem* ptr) {imageBuffer = ptr; }

	// Inherited via CLKernel
	virtual void create() override;

	virtual cl_event update() override;

	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;

	virtual void destroy() override;

};

