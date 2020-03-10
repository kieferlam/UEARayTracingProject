#pragma once
#include "CLKernel.h"
#include "RARKernel.h"
#include "ImageResolverKernel.h"

class RayTraceKernel :
	public CLKernel {

	// Ray Trace Stuff
	RayConfig* config;
	cl_mem configBuffer;

	World* world;

	cl_event updateEvent, queueEvent;

	cl_mem* rayBuffer;
	cl_mem* rayConfig;


	// Image Stuff
	ImageConfig imageConfig;
	cl_mem imageConfigBuffer;

	unsigned char* skyboxImages[6] = { nullptr ,nullptr ,nullptr ,nullptr ,nullptr ,nullptr };
	cl_mem skyboxBuffer;

	GLuint texture;
	cl_mem outputImageBuffer;

public:
	RayTraceKernel();
	~RayTraceKernel();

	// Ray trace methods
	inline void setPrimaryConfig(RayConfig* config_ptr) { config = config_ptr; }

	inline void setWorldPtr(World* ptr) { world = ptr; }

	inline cl_mem* getConfigBuffer() { return &configBuffer; }

	// Image methods
	inline void setTexture(GLuint t) { texture = t; }
	inline void setResolution(int w, int h) { imageConfig.res.x = w; imageConfig.res.y = h; };

	// Inherited via CLKernel
	virtual void create() override;
	virtual cl_event update() override;
	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;
	virtual void destroy() override;
};

