#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include <time.h>
#include <stdio.h>
#include "CLKernel.h"
#include "cl_helper.h"
#include "RARKernel.h"

__declspec (align(16)) struct ImageConfig {
	cl_int2 skyboxSize;
	cl_int2 res;
};

class ImageResolverKernel : public CLKernel {

	cl_mem* rayBuffer;
	cl_mem* rayConfig;
	cl_mem* materialBuffer;

	ImageConfig config;
	cl_mem configBuffer;

	unsigned char* skyboxImages[6] = { nullptr ,nullptr ,nullptr ,nullptr ,nullptr ,nullptr };
	cl_mem skyboxBuffer;

	GLuint texture;
	cl_mem outputImageBuffer;

	cl_event updateEvent;
	cl_event queueEvent;

public:
	ImageResolverKernel();
	~ImageResolverKernel();

	inline void setRayBuffer(cl_mem* ptr) { rayBuffer = ptr; }
	inline void setTexture(GLuint t) { texture = t; }
	inline void setResolution(int w, int h) { config.res.x = w; config.res.y = h; };

	inline void setMaterialBuffer(cl_mem* ptr) { materialBuffer = ptr; }

	inline void setRayConfig(cl_mem* ptr) { rayConfig = ptr; }

	virtual void create() override;

	virtual cl_event update() override;

	virtual void destroy() override;

	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;

};

