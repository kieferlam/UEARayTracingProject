#pragma once
#include <CL/opencl.h>
#include <string>
#include <glad/glad.h>
#include <time.h>
#include <stdio.h>
#include "CLKernel.h"
#include "cl_helper.h"
#include "World.h"
#include "Material.h"

__declspec (align(16)) struct Ray{
	cl_float3 origin;
	cl_float3 direction;
	cl_int2 pixelCoord;
} ;

__declspec (align(16)) struct RayConfig {
	cl_float aspect;
	cl_float width;
	cl_float height;
	cl_float screenDistance;
	cl_float3 camera;
	cl_int bounces;
};

__declspec (align(16)) struct TraceResult {
	Material material;
	Ray ray;
	cl_float3 intersect;
	cl_float3 normal;
	cl_float T;
	cl_float T2;
	cl_float cosine;
	cl_int sphereIndex;
	cl_int bounce;
	cl_bool hasIntersect;
	cl_bool hasTraced;
};

class RARKernel : public CLKernel {

	RayConfig* config;
	cl_mem configBuffer;

	TraceResult * results;
	cl_mem outputBuffer;

	World* world;

	cl_event updateEvent, queueEvent;

public:
	RARKernel();
	~RARKernel();

	inline void setPrimaryConfig(RayConfig* config_ptr) { config = config_ptr; }

	inline void setWorldPtr(World* ptr) { world = ptr; }

	inline cl_mem* getConfigBuffer() { return &configBuffer; }
	inline cl_mem* getRayBuffer() { return &outputBuffer; }

	void read();

	virtual void create() override;

	virtual cl_event update() override;

	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;

	virtual void destroy() override;

};

