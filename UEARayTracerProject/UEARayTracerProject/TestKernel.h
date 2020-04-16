#pragma once
#include "CLKernel.h"
#include <sstream>
#include <string>

class TestKernel :
	public CLKernel
{

	ModelStruct in_model;
	ModelStruct out_model;
	cl_mem in_modelBuffer;
	cl_mem out_modelBuffer;

	WorldStruct in_world;
	WorldStruct out_world;
	cl_mem in_worldBuffer;
	cl_mem out_worldBuffer;

	std::stringstream log;

	cl_event mainQueueEvent;

public:
	TestKernel();
	~TestKernel();

	inline void setModel(const ModelStruct& model) { in_model = model; }
	inline void setWorld(const WorldStruct& world) { in_world = world; }

	inline std::string getTestLog() { return log.str(); }

	// Inherited via CLKernel
	virtual void create() override;
	virtual cl_event update() override;
	virtual cl_event queue(cl_uint num_events, cl_event* wait_events) override;
	virtual void destroy() override;
};

