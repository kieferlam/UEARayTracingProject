#include "Mesh.h"
#include <algorithm>
#include "World.h"

constexpr cl_float3 X_AXIS = { 1.0f, 0.0f, 0.0f };
constexpr cl_float3 Y_AXIS = { 1.0f, 0.0f, 0.0f };
constexpr cl_float3 Z_AXIS = { 1.0f, 0.0f, 0.0f };

cl_float _mesh_dot(cl_float3 a, cl_float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

cl_float _mesh_projectToAxis(cl_float3 axis, cl_float3 point) {
	return _mesh_dot(point, axis);
}

cl_float3 _mesh_computeTriangleNormal(cl_float3 edge1, cl_float3 edge2) {
	return _world_normalise(_world_cross(edge1, edge2));
}

void getTriangleAxis(cl_float3* axis, cl_float3 v1, cl_float3 v2, cl_float3 v3) {
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

cl_float _mesh_dot(cl_float3 a, cl_float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

cl_float _mesh_projectVertex(cl_float3 plane, cl_float3 vertex) {
	return _mesh_dot(plane, vertex);
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

void Mesh::getTrianglesInGridCell(cl_float2* bounds, int cellIndex, cl_uint* cellTriangles, cl_uchar* triangleCount)
{

}
