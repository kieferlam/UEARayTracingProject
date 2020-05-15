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

#define NUM_RAY_CHILDREN (3)

__declspec (align(16)) struct Ray{
	cl_float3 origin;
	cl_float3 direction;
} ;

__declspec (align(16)) struct RayConfig {
	cl_float aspect;
	cl_float width;
	cl_float height;
	cl_float screenDistance;
	cl_float3 camera;
	cl_float pitch;
	cl_float yaw;
	cl_uint bounces;
};

__declspec (align(16)) struct TraceResult {
	Ray ray;

	cl_float3 intersect;

	cl_float3 normal;

	cl_float T;
	cl_float T2;
	cl_float cosine;
	cl_float shadowSoftness;

	cl_uint objectType;
	cl_uint objectIndex;
	cl_uint material;
	cl_uint bounce;

	cl_uint rayType;
	cl_uint pad1[3];

	cl_int hasIntersect;
	cl_int hasTraced;
	cl_int pad2[2];
};

class RARKernel : public CLKernel {

	RayConfig* config;
	cl_mem configBuffer;

	TraceResult * results;
	cl_mem outputBuffer;

	World* world;

	cl_mem* vertexBuffer;
	cl_mem* materialBuffer;
	cl_mem* sphereBuffer;
	cl_mem* triangleBuffer;
	cl_mem* modelBuffer;
	cl_mem* triangleGridBuffer;
	cl_mem* triangleCountGridBuffer;

	cl_event updateEvent, queueEvent;

public:
	RARKernel();
	~RARKernel();

	inline void setPrimaryConfig(RayConfig* config_ptr) { config = config_ptr; }

	inline void setWorldPtr(World* ptr) { world = ptr; }

	inline cl_mem* getConfigBuffer() { return &configBuffer; }
	inline cl_mem* getRayBuffer() { return &outputBuffer; }

	inline void setVertexBuffer(cl_mem* ptr) { vertexBuffer = ptr; }
	inline void setMaterialBuffer(cl_mem* ptr) { materialBuffer = ptr; }
	inline void setSphereBuffer(cl_mem* ptr) { sphereBuffer = ptr; }
	inline void setTriangleBuffer(cl_mem* ptr) { triangleBuffer = ptr; }
	inline void setModelBuffer(cl_mem* ptr) { modelBuffer = ptr; }
	inline void setTriangleGridBuffer(cl_mem* ptr) { triangleGridBuffer = ptr; }
	inline void setTrianlgeCountGridBuffer(cl_mem* ptr) { triangleCountGridBuffer = ptr; }

	void read();

	virtual void create() override;

	virtual cl_event update() override;

	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;

	virtual void destroy() override;

};

