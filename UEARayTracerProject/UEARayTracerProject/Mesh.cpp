#include "Mesh.h"
#include <algorithm>

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
