#define _CRT_SECURE_NO_WARNINGS

#define CL_TARGET_OPENCL_VERSION 210

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stddef.h>
#include <chrono>
#include "cl_helper.h"
#include "TracerKernel.h"
#include "RARKernel.h"
#include "ImageResolverKernel.h"
#include "CLKernel.h"
#include "World.h"
#include "RayTraceKernel.h"

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
CLKernel* kernels[] = {
	&rarkernel, &imagekernel
};

World world;
RayConfig config;

GLuint outputTexture;
GLuint shaderProgram;

GLuint vao;
GLuint quadVBO;

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
		 //"cl_kernels/raytrace.cl",
		 "cl_kernels/rarkernel.cl",
		 "cl_kernels/imageresolver.cl"
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
		
	});
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
	config.bounces = 1;

	int material = world.addMaterial({{ 0.3f, 0.4f, 0.5f }, 0.5f, 0.5f, 1.517f});

	int t1 = world.addVertex({ 0.0f, 0.0f, 40.0f });
	int t2 = world.addVertex({ 0.0f, 3.0f, 40.0f });
	int t3 = world.addVertex({ 3.0f, 3.0f, 40.0f });

	int tri = world.addTriangle(t1, t2, t3);

	world.setTriangleMaterial(tri, material);

	world.create();
	// Add spheres to world
	int sphere1 = world.addSphere({ -20.0f, -5.0f, 50.0f }, 10.0f, material);
	world.addSphere({ 20.0f, 5.0f, 50.0f }, 10.0f, material);
	world.update();

	rarkernel.setWorldPtr(&world);
	rarkernel.setPrimaryConfig(&config);
	rarkernel.setVertexBuffer(world.getVertexBufferPtr());
	rarkernel.setMaterialBuffer(world.getMaterialBufferPtr());

	imagekernel.setRayBuffer(rarkernel.getRayBuffer());
	imagekernel.setResolution(IMAGE_WIDTH, IMAGE_HEIGHT);
	imagekernel.setTexture(outputTexture);
	imagekernel.setRayConfig(rarkernel.getConfigBuffer());
	imagekernel.setMaterialBuffer(world.getMaterialBufferPtr());

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
	auto starttime = std::chrono::system_clock::now();

	cl_event worldUpdateEvent = NULL, rarEvent = NULL, imageEvent = NULL;
	cl_int worldUpdateStatus = -1, rarStatus = -1, imageStatus = -1;

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Time
		std::chrono::duration<double> elapsed_time = std::chrono::system_clock::now() - starttime;

		world.getSphere(sphere1)->position.x = 20.0f + cos(elapsed_time.count()) * 20.0f;
		world.getSphere(sphere1)->position.z = 100.0f + sin(elapsed_time.count()) * 10.0f;

		worldUpdateEvent = world.updateSpheres(sphere1, 1);

		rarEvent = rarkernel.queue(0, NULL);
		imageEvent = imagekernel.queue(1, &rarEvent);
		clFinish(cl::queue);

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