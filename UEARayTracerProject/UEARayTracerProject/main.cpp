#define _CRT_SECURE_NO_WARNINGS

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "cl_helper.h"
#include "TracerKernel.h"

#define WINDOW_WIDTH (1280)
#define WINDOW_HEIGHT (720)
#define IMAGE_WIDTH (1280)
#define IMAGE_HEIGHT (720)

GLFWwindow* window;

TracerKernel kernel;

GLuint outputTexture;

void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""), type, severity, message);
}

std::string readFile(const std::string& path) {
	std::ifstream file(path);
	return std::string(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
}

bool initGL() {
	// Initialise GLFW
	if (!glfwInit()) {
		std::cout << "Failed to initialise GLFW." << std::endl;
		return false;
	}

	// Create window for OpenGL version 4.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
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
#endif

	return true;
}

bool buildCL() {
	// Add sources
	std::string tracerSrc = readFile("cl_kernels/tracer.cl");
	cl::addSource(tracerSrc);

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

int main(void) {
	
	if (!initGL()) {
		std::cout << "Failed to initialise OpenGL." << std::endl;
		return -1;
	}

	if (!cl::init(true)) {
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

	// Create the kernel
	if (!kernel.create(outputTexture)) {
		std::cout << "Failed to create kernel." << std::endl;
		glfwTerminate();
		return -1;
	}

	return 0;
}