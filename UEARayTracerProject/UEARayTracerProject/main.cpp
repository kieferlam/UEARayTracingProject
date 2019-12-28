#define _CRT_SECURE_NO_WARNINGS

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stddef.h>
#include "cl_helper.h"
#include "TracerKernel.h"

#define WINDOW_WIDTH (1280)
#define WINDOW_HEIGHT (720)
#define IMAGE_WIDTH (1280)
#define IMAGE_HEIGHT (720)

#define VERTEX_SHADER_PATH ("shader/vertexshader.vert")
#define FRAGMENT_SHADER_PATH ("shader/fragmentshader.frag")
#define SAMPLER_UNIFORM ("textureSampler")

GLFWwindow* window;

TracerKernel kernel;

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

	// Create the kernel
	if (!kernel.create(outputTexture, IMAGE_WIDTH, IMAGE_HEIGHT)) {
		std::cout << "Failed to create kernel." << std::endl;
		glfwTerminate();
		return -1;
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

	// Send data
	kernel.writeKernelInput(CL_TRUE);

	// Main loop
	while (!glfwWindowShouldClose(window)) {

		// Trace
		kernel.trace(CL_TRUE);

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