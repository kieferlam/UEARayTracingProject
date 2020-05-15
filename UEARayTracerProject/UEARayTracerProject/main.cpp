#define _CRT_SECURE_NO_WARNINGS

#define CL_TARGET_OPENCL_VERSION 220

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stddef.h>
#include <chrono>
#include <random>
#include "cl_helper.h"
#include "TracerKernel.h"
#include "RARKernel.h"
#include "ImageResolverKernel.h"
#include "CLKernel.h"
#include "World.h"
#include "RayTraceKernel.h"
#include "ResetKernel.h"
#include "Model.h"
#include "TestKernel.h"
#include "ClearImageKernel.h"

constexpr float PI = 3.14159265359f;
constexpr float PI2 = 3.14159265359f * 2;
constexpr float HPI = 3.14159265359f * 0.5f;

#define WINDOW_WIDTH (1280)
#define WINDOW_HEIGHT (720)
#define IMAGE_WIDTH (1280)
#define IMAGE_HEIGHT (720)

#define VERTEX_SHADER_PATH ("shader/vertexshader.vert")
#define FRAGMENT_SHADER_PATH ("shader/fragmentshader.frag")
#define SAMPLER_UNIFORM ("textureSampler")

GLFWwindow* window;

RARKernel rarkernel;
ImageResolverKernel imagekernel;
RayTraceKernel raytracekernel;
ResetKernel resetkernel;
TestKernel testkernel;
ClearImageKernel clearimagekernel;
CLKernel* kernels[] = {
	&rarkernel, &imagekernel, &resetkernel, &testkernel, &clearimagekernel
};

World world;
RayConfig config;

GLuint outputTexture;
GLuint shaderProgram;

GLuint vao;
GLuint quadVBO;

const float kbdCameraSpeed = 1.0f;
bool yawRight, yawLeft;
bool pitchUp, pitchDown;

bool moveRight, moveLeft;
bool moveForward, moveBackward;
bool moveUp, moveDown;

bool benchmark_running;
double benchmark_start_time;
double benchmark_trace_time;
double benchmark_image_time;
std::vector<double> benchmark_trace, benchmark_image;
constexpr double BENCHMARK_TIME = 60.0;

const float cameraMoveSpeed = 20.0f;
const float mouseCameraSensitivity = 0.001f;
double mousex, mousey, oldmousex, oldmousey;

void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity, message);
}

void error_callback(int code, const char* description) {
	std::cout << "GLFW CALLBACK: " << code << " " << description << std::endl;
}

std::string readFile(const std::string path) {
	std::ifstream file(path, std::ios::in);
	return std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
}

bool initGL() {
	// Initialise GLFW
	if (!glfwInit()) {
		std::cout << "Failed to initialise GLFW." << std::endl;
		return false;
	}

	// Create window for OpenGL version 4.3
	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "UEA 3rd Year Project: Real-time Raytracing", NULL, NULL);

	// If window fails to load, terminate.
	if (!window) {
		glfwTerminate();
		std::cout << "Could not create GLFW window." << std::endl;
		return false;
	}

	glfwMakeContextCurrent(window);

	if (!gladLoadGL()) {
		glfwTerminate();
		std::cout << "Could not load Glad." << std::endl;
		return false;
	}

#if _DEBUG
	// Setup OpenGL error callback
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, 0);

	glfwSetErrorCallback(error_callback);
#endif


	return true;
}

bool buildCL() {
	const std::vector<std::string> sources = {
		 "cl_kernels/kernels.cl"
	};

	for (auto it = sources.begin(); it != sources.end(); ++it) {
		cl::addSource(readFile(*it));
	}

	// Build
	if (!cl::build()) {
		return false;
	}
	return true;
}

GLuint createEmptyTexture(int width, int height) {
	GLuint handle;
	glGenTextures(1, &handle);

	glBindTexture(GL_TEXTURE_2D, handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_FLOAT, NULL);

	return handle;
}

void glPrintShaderLog(GLuint shader) {
	GLint logLength;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

	char* log = new char[logLength];
	glGetShaderInfoLog(shader, logLength, NULL, log);

	std::cout << "Shader compile error:" << std::endl;
	std::cout << log << std::endl;

	delete[] log;
}

void glPrintProgramLog(GLuint program) {
	GLint logLength;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

	char* log = new char[logLength];
	glGetProgramInfoLog(program, logLength, NULL, log);

	std::cout << "Program compile error:" << std::endl;
	std::cout << log << std::endl;

	delete[] log;
}

GLuint createShaderProgram() {
	GLuint program = glCreateProgram();
	GLuint vshader, fshader;
	vshader = glCreateShader(GL_VERTEX_SHADER);
	fshader = glCreateShader(GL_FRAGMENT_SHADER);

	// Load shader sources
	std::string vshaderStr = readFile(VERTEX_SHADER_PATH);
	const char* vshaderCstr = vshaderStr.c_str();
	const GLint vshaderLen = vshaderStr.length();
	glShaderSource(vshader, 1, &vshaderCstr, &vshaderLen);

	std::string fshaderStr = readFile(FRAGMENT_SHADER_PATH);
	const char* fshaderCstr = fshaderStr.c_str();
	const GLint fshaderLen = fshaderStr.length();
	glShaderSource(fshader, 1, &fshaderCstr, &fshaderLen);

	bool hasError = false;
	GLint compileStatus;

	// Compile shaders
	glCompileShader(vshader);
	glGetShaderiv(vshader, GL_COMPILE_STATUS, &compileStatus);
	hasError |= compileStatus == GL_FALSE;
	if (compileStatus == GL_FALSE) {
		glPrintShaderLog(vshader);
	}

	glCompileShader(fshader);
	glGetShaderiv(fshader, GL_COMPILE_STATUS, &compileStatus);
	hasError |= compileStatus == GL_FALSE;
	if (compileStatus == GL_FALSE) {
		glPrintShaderLog(fshader);
	}

	// Error reporting
	if (hasError) {
		std::cout << "Error creating shaders." << std::endl;
		glDeleteShader(vshader);
		glDeleteShader(fshader);
		glDeleteProgram(program);
		return 0;
	}

	// Attach shaders to program and link
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);

	glLinkProgram(program);

	// Detach and delete as it is no longer needed with the program already linked
	glDetachShader(program, vshader);
	glDetachShader(program, fshader);
	glDeleteShader(vshader);
	glDeleteShader(fshader);

	// Error checking for shader program
	GLint linkStatus;
	glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
	if (linkStatus == GL_FALSE) {
		glPrintProgramLog(program);
		glDeleteProgram(program);
		return 0;
	}

	return program;
}

void createRenderQuad() {
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &quadVBO);

	const float vertices[] = {
		-1.0f, -1.0f, 0.0f, 1.0f, // BOTTOM LEFT
		1.0f, -1.0f, 1.0f, 1.0f, // BOTTOM RIGHT
		1.0f, 1.0f, 1.0f, 0.0f, // TOP RIGHT
		-1.0f, -1.0f, 0.0f, 1.0f, // BOTTOM LEFT
		1.0f, 1.0f, 1.0f, 0.0f, // TOP RIGHT
		-1.0f, 1.0f, 0.0f, 0.0f, // TOP LEFT
	};

	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, false, sizeof(GLfloat) * 4, 0);
	glVertexAttribPointer(1, 2, GL_FLOAT, false, sizeof(GLfloat) * 4, (void*)(sizeof(GLfloat) * 2));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}

void setupEventHandlers() {
	glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (key == GLFW_KEY_ESCAPE) glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		switch (key) {
		case GLFW_KEY_RIGHT:
			yawRight = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_LEFT:
			yawLeft = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_UP:
			pitchUp = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_DOWN:
			pitchDown = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_W:
			moveForward = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_S:
			moveBackward = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_D:
			moveRight = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_A:
			moveLeft = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_SPACE:
			moveUp = action != GLFW_RELEASE ? true : false;
			break;
		case GLFW_KEY_LEFT_SHIFT:
			moveDown = action != GLFW_RELEASE ? true : false;
			break;

			// Testing
		case GLFW_KEY_0:
			if (benchmark_running) break;
			std::cout << "Starting benchmark." << std::endl;
			benchmark_running = true;
			benchmark_start_time = glfwGetTime();
			benchmark_trace.clear();
			benchmark_image.clear();
			break;
		}
	});
	glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	});
	glfwGetCursorPos(window, &oldmousex, &oldmousey);
	mousex = oldmousex;
	mousey = oldmousey;
	glfwSetCursorPosCallback(window, [](GLFWwindow* window, double xpos, double ypos) {
		oldmousex = mousex;
		oldmousey = mousey;
		mousex = xpos;
		mousey = ypos;

		if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
			config.yaw += (mousex - oldmousex) * mouseCameraSensitivity;
			config.pitch += (mousey - oldmousey) * mouseCameraSensitivity;
		}
	});
}

void updateCameraMovement(float deltaTime) {
	if (yawRight) config.yaw += deltaTime * kbdCameraSpeed;
	if (yawLeft) config.yaw -= deltaTime * kbdCameraSpeed;
	if (pitchUp) config.pitch += deltaTime * kbdCameraSpeed;
	if (pitchDown) config.pitch -= deltaTime * kbdCameraSpeed;
	// Normalise
	if (config.yaw > PI2) config.yaw -= PI2;
	if (config.yaw < 0.0f) config.yaw += PI2;
	if (config.pitch > HPI) config.pitch = HPI;
	if (config.pitch < -HPI) config.pitch = -HPI;

	// Movement
	if (moveForward) config.camera = { config.camera.x + sin(config.yaw) * cameraMoveSpeed * deltaTime, config.camera.y, config.camera.z + cos(config.yaw) * cameraMoveSpeed * deltaTime };
	if (moveBackward) config.camera = { config.camera.x - sin(config.yaw) * cameraMoveSpeed * deltaTime, config.camera.y, config.camera.z - cos(config.yaw) * cameraMoveSpeed * deltaTime };
	if (moveRight) config.camera = { config.camera.x + sin(config.yaw + HPI) * cameraMoveSpeed * deltaTime, config.camera.y, config.camera.z + cos(config.yaw + HPI) * cameraMoveSpeed * deltaTime };
	if (moveLeft) config.camera = { config.camera.x + sin(config.yaw - HPI) * cameraMoveSpeed * deltaTime, config.camera.y, config.camera.z + cos(config.yaw - HPI) * cameraMoveSpeed * deltaTime };
	if (moveUp) config.camera.y += cameraMoveSpeed * deltaTime;
	if (moveDown) config.camera.y -= cameraMoveSpeed * deltaTime;
}

void runStructChecks() {
	std::cout << "Size of cl_char:\t" << sizeof(cl_char) << std::endl;
	std::cout << "Size of cl_int:\t" << sizeof(cl_int) << std::endl;
	std::cout << "Size of cl_float:\t" << sizeof(cl_float) << std::endl;
	std::cout << "Size of cl_float2:\t" << sizeof(cl_float2) << std::endl;
	std::cout << "Size of cl_float3:\t" << sizeof(cl_float3) << std::endl;
	std::cout << "Size of cl_float4:\t" << sizeof(cl_float4) << std::endl;

	std::cout << "Size of ModelStruct:\t" << sizeof(ModelStruct) << "\tr.16:\t" << sizeof(ModelStruct) % 16 << std::endl;
	std::cout << "Size of WorldStruct:\t" << sizeof(WorldStruct) << "\tr.16:\t" << sizeof(WorldStruct) % 16 << std::endl;
	std::cout << "Size of RayConfig:\t" << sizeof(RayConfig) << "\tr.16:\t" << sizeof(RayConfig) % 16 << std::endl;
	std::cout << "Size of Material:\t" << sizeof(Material) << "\tr.16:\t" << sizeof(Material) % 16 << std::endl;
	std::cout << "Size of Sphere:\t\t" << sizeof(Sphere) << "\tr.16:\t" << sizeof(Sphere) % 16 << std::endl;
	std::cout << "Size of Triangle:\t" << sizeof(Triangle) << "\tr.16:\t" << sizeof(Triangle) % 16 << std::endl;
	std::cout << "Size of TraceResult:\t" << sizeof(TraceResult) << "\tr.16:\t" << sizeof(TraceResult) % 16 << std::endl;
	std::cout << "Size of ImageConfig:\t" << sizeof(ImageConfig) << "\tr.16:\t" << sizeof(ImageConfig) % 16 << std::endl;

	std::cout << "ModelStruct members" << std::endl;
	std::cout << "triangleGridOffset\t" << sizeof(ModelStruct().triangleGridOffset) << "\tr.16\t" << sizeof(ModelStruct().triangleGridOffset) % 16 << std::endl;
	std::cout << "triangleCountOffset\t" << sizeof(ModelStruct().triangleCountOffset) << "\tr.16\t" << sizeof(ModelStruct().triangleCountOffset) % 16 << std::endl;
	std::cout << "bounds\t\t\t" << sizeof(ModelStruct().bounds) << "\tr.16\t" << sizeof(ModelStruct().bounds) % 16 << std::endl;
	std::cout << "triangleOffset\t\t" << sizeof(ModelStruct().triangleOffset) << "\tr.16\t" << sizeof(ModelStruct().triangleOffset) % 16 << std::endl;
	std::cout << "numTriangles\t\t" << sizeof(ModelStruct().numTriangles) << "\tr.16\t" << sizeof(ModelStruct().numTriangles) % 16 << std::endl;

	std::cout << "WorldStruct members" << std::endl;
	std::cout << "numSpheres\t" << sizeof(WorldStruct().numSpheres) << "\tr.16\t" << sizeof(WorldStruct().numSpheres) % 16 << std::endl;
	std::cout << "numTriangles\t" << sizeof(WorldStruct().numTriangles) << "\tr.16\t" << sizeof(WorldStruct().numTriangles) % 16 << std::endl;
	std::cout << "numModels\t" << sizeof(WorldStruct().numModels) << "\tr.16\t" << sizeof(WorldStruct().numModels) % 16 << std::endl;
}

void runKernelTest(ModelStruct* mstruct, WorldStruct* wstruct) {
	ModelStruct in_model;
	if (mstruct == nullptr) {
		in_model.triangleGridOffset = 3;
		in_model.triangleCountOffset = 6;
		for (int i = 0; i < 7; ++i) {
			in_model.bounds[i] = { (float)i * 2 + 1, (float)i * 2 + 2 };
		}
		in_model.triangleOffset = 5;
		in_model.numTriangles = 10;
	} else {
		in_model = *mstruct;
	}

	WorldStruct in_world;
	if (wstruct == nullptr) {
		in_world.numSpheres = 6;
		in_world.numTriangles = 7;
		in_world.numModels = 8;
	} else {
		in_world = *wstruct;
	}

	testkernel.setModel(in_model);
	testkernel.setWorld(in_world);
	cl_event wait = testkernel.update();
	testkernel.queue(1, &wait);

	std::string log = testkernel.getTestLog();
	
	std::cout << "Test kernel log size(" << log.length() << "): " << std::endl;
	std::cout << log << std::endl;
}

void save_benchmark() {
	std::cout << "Saving benchmark." << std::endl;
	std::ofstream benchmark_file;
	benchmark_file.open("benchmark.txt");
	benchmark_file << "trace,image" << std::endl;
	for (int i = 0; i < benchmark_trace.size(); ++i) {
		benchmark_file << benchmark_trace[i] << "," << benchmark_image[i] << std::endl;
	}
	benchmark_file.close();
}

void testscene() {
	Material m;
	m.diffuse = {1.0f, 1.0f, 1.0f};
	m.opacity = 1.0f;
	m.reflectivity = 1.0f;
	m.refractiveIndex = 1.517f;
	m.specular = 100.0f;
	int material = world.addMaterial(m);

	int diffuse[100];
	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> range(0.0f, 1.0f);
	for (int i = 0; i < sizeof(diffuse) / sizeof(diffuse[0]); ++i) {
		diffuse[i] = world.addMaterial({ { range(rng) * 0.4f + 0.2f, range(rng) * 0.4f + 0.2f, range(rng) * 0.4f + 0.2f }, range(rng) * 1000.0f, 0.0f, 1.0f, 1.0f });
		float rad = range(rng) * 5.0f + 1.0f;
		world.addSphere({ range(rng) * 300.0f - 100.0f, rad, range(rng) * 100.0f }, rad, diffuse[i]);
	}

	int ms[10];
	for (int i = 0; i < 10; ++i) {
		ms[i] = world.addMaterial({ {1.0f, 1.0f, 1.0f}, 1000.0f, 1.0f, 0.0f, 1.0f + i * 0.07f });
		world.addSphere({ i * 10.0f, 5.0f, -50.0f }, 5.0f, ms[i]);
	}

	// Floor
	float floorrad = 50000.0f;
	int floormat = world.addMaterial({ {132.0f / 255.0f, 153.0f / 255.0f, 179.0f / 255.0f}, 0.0f, 0.0f, 1.0f, 1.0f });
	world.addSphere({ 0.0f, -floorrad, 0.0f }, floorrad, floormat);
}

void scene1() {
	float specular = 1000.0f;
	int mirror = world.addMaterial({ {1.0f, 1.0f, 1.0f}, specular, 1.0f, 1.0f, 1.0f });
	int hollowglass = world.addMaterial({ { 1.0f, 1.0f, 1.0f }, specular, 1.0f, 0.0f, 1.04f });
	int solidglass = world.addMaterial({ { 1.0f, 1.0f, 1.0f }, specular, 1.0f, 0.0f, 1.517f });
	int floormat = world.addMaterial({ {132.0f / 255.0f, 153.0f / 255.0f, 179.0f / 255.0f}, specular, 0.0f, 1.0f, 1.0f });
	int diffuse[10];
	std::default_random_engine rng(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<float> range(0.0f, 1.0f);
	for (int i = 0; i < sizeof(diffuse) / sizeof(diffuse[0]); ++i) {
		diffuse[i] = world.addMaterial({{ range(rng) * 0.4f + 0.2f, range(rng) * 0.4f + 0.2f, range(rng) * 0.4f + 0.2f }, range(rng) * 1000.0f, 0.0f, 1.0f, 1.0f});
	}

	world.addSphere({ 10.0f, 40.0f, -30.0f }, 40.0f, 0);
	world.addSphere({ 300.0f, 40.0f, 60.0f }, 40.0f, 1);
	world.addSphere({ 50.0f, 40.0f, 160.0f }, 40.0f, 2);

	for (int i = 0; i < 15; ++i) {
		float rad = range(rng) * 9.0f + 1.0f;
		world.addSphere({range(rng) * 200.0f, rad + 1.0f, range(rng) * 200.0f}, rad, (int)(range(rng)*13));
	}

	// Floor
	float floorrad = 50000.0f;
	world.addSphere({ 0.0f, -floorrad, 0.0f }, floorrad, floormat);
}

void scene2() {
	int material = world.addMaterial({ { 1.0f, 1.0f, 1.0f }, 100.0f, 1.0f, 1.0f, 1.04f });
	int material2 = world.addMaterial({ { 1.0f, 1.0f, 1.0f }, 100.0f, 0.0f, 1.0f, 1.0f });
	int material3 = world.addMaterial({ { 1.0f, 1.0f, 1.0f }, 100.0f, 0.0f, 1.0f, 1.3f });

	Model testModel;
	testModel.loadFromFile("data/monkey.obj", &world, 10.0f);
}

void reflection_scene() {
	int reflect = world.addMaterial({ {1.0f, 1.0f, 1.0f}, 100.0f, 1.0f, 1.0f, 1.0f });

	world.addSphere({ 0.0f, 10.0f, 0.0f }, 10.0f, reflect);
	world.addSphere({ 25.0f, 10.0f, 0.0f }, 10.0f, reflect);
}

void benchmark_scene_baseline() {

}

void benchmark_scene_spheres(int spheres) {
	std::default_random_engine rng;
	std::uniform_real_distribution<float> range(0.0f, 1.0f);

	float reflect = 0.0f;
	float opacity = 1.0f;
	float refractiveIndex = 1.0f;
	float rad = 10.0f;
	for (int i = 0; i < spheres; ++i) {
		int mat = world.addMaterial({ {range(rng) * 0.4f + 0.4f, range(rng) * 0.4f + 0.4f, range(rng) * 0.4f + 0.4f}, range(rng)*1000.0f + 1.0f, reflect, opacity, refractiveIndex });
		world.addSphere({ (range(rng) - 0.5f) * 500.0f, (range(rng) - 0.5f) * 100.0f, range(rng) * 500.0f + 50.0f}, rad, mat);
	}
}

int main(void) {

	if (!initGL()) {
		std::cout << "Failed to initialise OpenGL." << std::endl;
		return -1;
	}

	if (!cl::init()) {
		std::cout << "Failed to initialise OpenCL." << std::endl;
		glfwTerminate();
		return -1;
	}

	// Load and build OpenCL kernel sources
	if (!buildCL()) {
		std::cout << "Could not build OpenCL program." << std::endl;
		glfwTerminate();
		return -1;
	}

	// Create empty texture for kernel output
	outputTexture = createEmptyTexture(IMAGE_WIDTH, IMAGE_HEIGHT);

	// Config
	config.camera = { 0.0f, 0.0f, 0.0f };
	config.screenDistance = 2.0f;
	config.aspect = (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT;
	config.width = WINDOW_WIDTH;
	config.height = WINDOW_HEIGHT;
	config.bounces = 2;

	//testscene();
	//reflection_scene();
	//scene1();
	scene2();
	//benchmark_scene_baseline();
	//benchmark_scene_spheres(100);

	world.create();

	rarkernel.setWorldPtr(&world);
	rarkernel.setPrimaryConfig(&config);
	rarkernel.setVertexBuffer(world.getVertexBufferPtr());
	rarkernel.setMaterialBuffer(world.getMaterialBufferPtr());
	rarkernel.setSphereBuffer(world.getSphereBufferPtr());
	rarkernel.setTriangleBuffer(world.getTriangleBufferPtr());
	rarkernel.setModelBuffer(world.getModelBufferPtr());
	rarkernel.setTriangleGridBuffer(world.getTriangleGridPtr());
	rarkernel.setTrianlgeCountGridBuffer(world.getTriangleCountPtr());

	imagekernel.setRayBuffer(rarkernel.getRayBuffer());
	imagekernel.setResolution(IMAGE_WIDTH, IMAGE_HEIGHT);
	imagekernel.setTexture(outputTexture);
	imagekernel.setRayConfig(rarkernel.getConfigBuffer());
	imagekernel.setMaterialBuffer(world.getMaterialBufferPtr());

	resetkernel.setConfig(&config);
	resetkernel.setConfigBuffer(rarkernel.getConfigBuffer());
	resetkernel.setRayBuffer(rarkernel.getRayBuffer());

	clearimagekernel.setImage(imagekernel.getImageBufferPtr());
	clearimagekernel.setImageConfig(imagekernel.getImageConfig());
	clearimagekernel.setImageConfigBuffer(imagekernel.getImageConfigBufferPtr());

	/*raytracekernel.setPrimaryConfig(&config);
	raytracekernel.setResolution(IMAGE_WIDTH, IMAGE_HEIGHT);
	raytracekernel.setTexture(outputTexture);
	raytracekernel.setWorldPtr(&world);*/

	// Create kernels
	for (int i = 0; i < sizeof(kernels) / sizeof(CLKernel*); ++i) {
		if (!kernels[i]->createKernel()) {
			std::cout << "Couldn't create kernel. Aborting." << std::endl;
			return -1;
		}
		kernels[i]->create();
		kernels[i]->update(); // Update once to upload data to buffers
	}

	// Run kernel struct test
	runStructChecks();
	runKernelTest(nullptr, nullptr);

	// Setup OpenGL for rendering
	// Create shader program
	shaderProgram = createShaderProgram();
	if (shaderProgram == 0) {
		std::cout << "Couldn't create shader program." << std::endl;
		return -1;
	}

	// Get uniforms
	GLint samplerUniformLoc = glGetUniformLocation(shaderProgram, SAMPLER_UNIFORM);

	// Setup render quad
	createRenderQuad();

	setupEventHandlers();

	// Time
	double starttime = glfwGetTime();
	double lastframetime = starttime;

	cl_event worldUpdateEvent = NULL, rarEvent = NULL, imageEvent = NULL, resetEvent = NULL, clearimgEvent = NULL;
	cl_int worldUpdateStatus = -1, rarStatus = -1, imageStatus = -1;

	std::default_random_engine reng;
	std::uniform_real_distribution<float> sphere_dist(0.0f, 1.0f);
	float* rands = new float[world.getSpheres().size()];
	for (int i = 0; i < world.getSpheres().size(); ++i) {
		rands[i] = sphere_dist(reng);
	}

	// Init benchmark stuff
	constexpr size_t benchmark_reserve_size = 10 ^ 6;
	benchmark_trace.reserve(benchmark_reserve_size);
	benchmark_image.reserve(benchmark_reserve_size);

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Time
		double now = glfwGetTime();
		float deltaTime = now - lastframetime;
		lastframetime = now;

		for (int i = 0; i < (int)world.getSpheres().size() - 1; ++i) {
			world.getSphere(i)->position.y = world.getSphere(i)->radius + abs(sin(now + rands[i])) * 5.0f;
		}

		//worldUpdateEvent = world.updateSpheres(0, world.getSpheres().size());

		if(!benchmark_running) updateCameraMovement(deltaTime);

		rarkernel.update();

		//clearimgEvent = clearimagekernel.queue(0, NULL);
		resetEvent = resetkernel.queue(0, NULL);

		if (benchmark_running) benchmark_trace_time = glfwGetTime();
		rarEvent = rarkernel.queue(1, &resetEvent);
		clWaitForEvents(1, &rarEvent);
		if (benchmark_running) benchmark_trace.push_back(glfwGetTime() - benchmark_trace_time);

		if (benchmark_running) benchmark_image_time = glfwGetTime();
		imageEvent = imagekernel.queue(1, &rarEvent);
		clWaitForEvents(1, &imageEvent);
		if (benchmark_running) benchmark_image.push_back(glfwGetTime() - benchmark_image_time);

		clFinish(cl::queue);

		if (benchmark_running) {
			if (glfwGetTime() - benchmark_start_time >= BENCHMARK_TIME) {
				benchmark_running = false;
				save_benchmark();
			}
		}

		// Render
		glUseProgram(shaderProgram);

		glUniform1i(samplerUniformLoc, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, outputTexture);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}