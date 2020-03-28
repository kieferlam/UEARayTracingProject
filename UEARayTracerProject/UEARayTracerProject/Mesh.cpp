#include "Mesh.h"
#include <algorithm>
#include "World.h"

constexpr cl_float3 X_AXIS = { 1.0f, 0.0f, 0.0f };
constexpr cl_float3 Y_AXIS = { 1.0f, 0.0f, 0.0f };
constexpr cl_float3 Z_AXIS = { 1.0f, 0.0f, 0.0f };

cl_float _mesh_dot(const cl_float3 & a, const cl_float3 & b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

cl_float _mesh_projectToAxis(const cl_float3 & axis, const cl_float3 & point) {
	return _mesh_dot(point, axis);
}

cl_float3 _mesh_computeTriangleNormal(const cl_float3 & edge1, const cl_float3 & edge2) {
	return _world_normalise(_world_cross(edge1, edge2));
}

void _mesh_getTriangleAxis(cl_float3* axis, const cl_float3 & v1, const cl_float3 & v2, const cl_float3 & v3) {
	cl_float3 edge1 = v2 - v1;
	cl_float3 edge2 = v3 - v1;
	cl_float3 edge3 = v2 - v3;
	cl_float3 normal = _mesh_computeTriangleNormal(edge1, edge2);
	axis[0] = edge1;
	axis[1] = edge2;
	axis[2] = edge3;
	axis[3] = normal;
	axis[4] = _world_cross(X_AXIS, edge1);
	axis[5] = _world_cross(Y_AXIS, edge1);
	axis[6] = _world_cross(Z_AXIS, edge1);
	axis[7] = _world_cross(X_AXIS, edge2);
	axis[8] = _world_cross(Y_AXIS, edge2);
	axis[9] = _world_cross(Z_AXIS, edge2);
}

Mesh::Mesh()
{
}

Mesh::~Mesh()
{
}

cl_float _mesh_min(cl_float* values, size_t count) {
	if (count == 0) {
		std::cout << "Warning: attempting to find min value of empty vector." << std::endl;
		return 0.0f;
	}
	cl_float min = values[0];
	for (int i = 1; i < count; ++i) {
		if (values[i] < min) {
			min = values[i];
		}
	}
	return min;
}

cl_float _mesh_max(cl_float* values, size_t count) {
	if (count == 0) {
		std::cout << "Warning: attempting to find max value of empty vector." << std::endl;
		return 0.0f;
	}
	cl_float max = values[0];
	for (int i = 1; i < count; ++i) {
		if (values[i] > max) {
			max = values[i];
		}
	}
	return max;
}

cl_float _mesh_projectVertex(cl_float3 plane, cl_float3 vertex) {
	return _mesh_dot(plane, vertex);
}

bool _mesh_overlap(cl_float a_min, cl_float a_max, cl_float b_min, cl_float b_max) {
	if (a_min > b_max || a_max < b_min) return false;
	return true;
}

bool _mesh_aabb(cl_float2 x1, cl_float2 y1, cl_float2 z1, cl_float2 x2, cl_float2 y2, cl_float2 z2) {
	if (!_mesh_overlap(x1.x, x1.y, x2.x, x2.y)) return false;
	if (!_mesh_overlap(y1.x, y1.y, y2.x, y2.y)) return false;
	if (!_mesh_overlap(z1.x, z1.y, z2.x, z2.y)) return false;
	return true;
}

void Mesh::createBoundingVolume(const Triangle* faces, const std::vector<cl_float3> & vertices)
{
	for (int i = 0; i < sizeof(BVH_PlaneNormals) / sizeof(BVH_PlaneNormals[0]); ++i) {
		cl_float3 planeNormal = BVH_PlaneNormals[i];

		// Find min max of vertex in plane normal

		bounds[i].x = std::numeric_limits<float>::max(); // Min
		bounds[i].y = std::numeric_limits<float>::min(); // Max

		for (auto it = triangles.begin(); it != triangles.end(); ++it) {
			const Triangle face = faces[*it];
			// Min
			bounds[i].x = std::min(bounds[i].x, _mesh_projectVertex(planeNormal, vertices[face.face.x]));
			bounds[i].x = std::min(bounds[i].x, _mesh_projectVertex(planeNormal, vertices[face.face.y]));
			bounds[i].x = std::min(bounds[i].x, _mesh_projectVertex(planeNormal, vertices[face.face.z]));
			// Max
			bounds[i].y = std::max(bounds[i].y, _mesh_projectVertex(planeNormal, vertices[face.face.x]));
			bounds[i].y = std::max(bounds[i].y, _mesh_projectVertex(planeNormal, vertices[face.face.y]));
			bounds[i].y = std::max(bounds[i].y, _mesh_projectVertex(planeNormal, vertices[face.face.z]));
		}

	}
}

/**
	This function adds the triangles from the mesh in each cell into the grid stored in the model struct.

	cl_float2 * bounds: This is an array of bounds in each dimension (X,Y,Z) of the grid cell where each bounds contains the min and max value in the x and y components respectively.
*/
void Mesh::getTrianglesInGridCell(World * world, cl_float2* bounds, cl_uint* cellTriangles, cl_uchar* triangleCount)
{
	cl_float3 axis[13] = { X_AXIS, Y_AXIS, Z_AXIS };

	for(auto tri = triangles.begin(); tri != triangles.end(); ++tri){
		Triangle* face = world->getTriangle(*tri);
		cl_float3 vertices[3] = { world->getVertexBuffer()[face->face.x], world->getVertexBuffer()[face->face.y], world->getVertexBuffer()[face->face.z] };

		if (*triangleCount >= GRID_MAX_TRIANGLES_PER_CELL) {
			std::cout << "Maximum triangles per grid cell reached!" << std::endl;
			return;
		}

		// If a vertex is in the cell, the triangle is in the cell
		bool skipTriangle = false;
		for (int i = 0; i < 3; ++i) {
			if (_mesh_aabb(bounds[0], bounds[1], bounds[2], { vertices[i].x, vertices[i].x }, { vertices[i].y, vertices[i].y }, { vertices[i].z, vertices[i].z })) {
				cellTriangles[*triangleCount++] = *tri;
				skipTriangle = true;
				break;
			}
		}
		if (skipTriangle) continue;

		// Perform SAT test for tri box
		_mesh_getTriangleAxis(&axis[3], vertices[0], vertices[1], vertices[2]);
		// Get box vertices
		cl_float3 boxVertices[6] = {
			{ bounds[0].x, bounds[1].x, bounds[2].x },
			{ bounds[0].y, bounds[1].y, bounds[2].y },
			{ bounds[0].y, bounds[1].y, bounds[2].x },
			{ bounds[0].y, bounds[1].x, bounds[2].x },
			{ bounds[0].x, bounds[1].x, bounds[2].y },
			{ bounds[0].x, bounds[1].y, bounds[2].y }
		};
		// Axis loop
		for (int i = 0; i < 12; ++i) {
			// Project box onto axis
			cl_float boxP[6];
			for (int b = 0; b < 6; ++b) {
				boxP[b] = _mesh_projectToAxis(axis[i], boxVertices[b]);
			}

			// Project triangle onto axis
			cl_float triP[3];
			for (int v = 0; v < 3; ++v) {
				triP[v] = _mesh_projectToAxis(axis[i], vertices[v]);
			}

			// If no overlap, continue to next triangle
			if (!_mesh_overlap(_mesh_min(boxP, 6), _mesh_max(boxP, 6), _mesh_min(triP, 3), _mesh_max(triP, 3))) {
				skipTriangle = true;
				break;
			}
		}
		if (skipTriangle) continue;

		// There is overlap in all axis so add triangle
		cellTriangles[*triangleCount++] = *tri;
	}
}
