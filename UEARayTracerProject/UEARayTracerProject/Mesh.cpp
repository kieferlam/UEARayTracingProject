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
	cl_float3 edge1 = _world_normalise(v2 - v1);
	cl_float3 edge2 = _world_normalise(v3 - v1);
	cl_float3 edge3 = _world_normalise(v2 - v3);
	cl_float3 normal = _mesh_computeTriangleNormal(edge1, edge2);
	axis[0] = edge1;
	axis[1] = edge2;
	axis[2] = edge3;
	axis[3] = normal;
	axis[4] = _world_normalise(_world_cross(X_AXIS, edge1));
	axis[5] = _world_normalise(_world_cross(Y_AXIS, edge1));
	axis[6] = _world_normalise(_world_cross(Z_AXIS, edge1));
	axis[7] = _world_normalise(_world_cross(X_AXIS, edge2));
	axis[8] = _world_normalise(_world_cross(Y_AXIS, edge2));
	axis[9] = _world_normalise(_world_cross(Z_AXIS, edge2));
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

		for (auto it = octree.triangles.begin(); it != octree.triangles.end(); ++it) {
			const Triangle face = faces[*it];
			cl_float projected[3] = {
				_mesh_projectVertex(planeNormal, vertices[face.face.x]),
				_mesh_projectVertex(planeNormal, vertices[face.face.y]),
				_mesh_projectVertex(planeNormal, vertices[face.face.z])
			};
			for (int b = 0; b < 3; ++b) {
				bounds[i].x = std::min(bounds[i].x, projected[b]);
				bounds[i].y = std::max(bounds[i].y, projected[b]);
			}
		}

	}
}

bool _mesh_triangleBoxIntersect(const cl_float3 * vertices, const cl_float2 * bounds) {
	cl_float3 axis[13] = { X_AXIS, Y_AXIS, Z_AXIS };

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
			return false;
		}
	}

	return true;
}

/**
	This function adds the triangles from the mesh in each cell into the grid stored in the model struct.

	cl_float2 * bounds: This is an array of bounds in each dimension (X,Y,Z) of the grid cell where each bounds contains the min and max value in the x and y components respectively.
*/
void Mesh::getTrianglesInGridCell(World * world, cl_float2* bounds, cl_uint* cellTriangles, cl_uchar* triangleCount)
{

	for(auto tri = octree.triangles.begin(); tri != octree.triangles.end(); ++tri){
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
				cellTriangles[(*triangleCount)++] = *tri;
				skipTriangle = true;
				break;
			}
		}
		if (skipTriangle) continue;

		// Perform SAT test for tri box
		if (!_mesh_triangleBoxIntersect(vertices, bounds)) continue;

		// There is overlap in all axis so add triangle
		cellTriangles[(*triangleCount)++] = *tri;
	}
}

void _mesh_createOctreeChildren(OctreeCell* parentCell, World* world, unsigned int maxDepth, std::vector<OctreeCell*> & leafnodes) {
	float midpoints[3];
	for (int i = 0; i < 3; ++i) midpoints[i] = (parentCell->bounds[i].y + parentCell->bounds[i].x) * 0.5f;

	// Initialize children
	for (int i = 0; i < 8; ++i) {
		parentCell->children[i] = new OctreeCell();
		parentCell->children[i]->depth = parentCell->depth + 1;
	}

	// Set bounds for children
	// -X-Y-Z
	for (int i = 0; i < 3; ++i) parentCell->children[0]->bounds[i] = { parentCell->bounds[i].x, midpoints[i] };
	// -X-Y+Z
	for(int i = 0; i < 2; ++i) parentCell->children[1]->bounds[i] = { parentCell->bounds[i].x, midpoints[i] };
	parentCell->children[1]->bounds[2] = { midpoints[2], parentCell->bounds[2].y };
	// -X+Y-Z
	parentCell->children[2]->bounds[0] = { parentCell->bounds[0].x, midpoints[0]};
	parentCell->children[2]->bounds[1] = { midpoints[1], parentCell->bounds[1].y };
	parentCell->children[2]->bounds[2] = { parentCell->bounds[2].x, midpoints[2] };
	// -X+Y+Z
	parentCell->children[3]->bounds[0] = { parentCell->bounds[0].x, midpoints[0] };
	for(int i = 1; i < 3; ++i) parentCell->children[3]->bounds[i] = { midpoints[i], parentCell->bounds[i].y };
	// +X-Y-Z
	parentCell->children[4]->bounds[0] = { midpoints[0], parentCell->bounds[0].y };
	for (int i = 1; i < 3; ++i) parentCell->children[4]->bounds[i] = { parentCell->bounds[i].x, midpoints[i] };
	// +X-Y+Z
	parentCell->children[5]->bounds[0] = { midpoints[0], parentCell->bounds[0].y };
	parentCell->children[5]->bounds[1] = { parentCell->bounds[1].x, midpoints[1] };
	parentCell->children[5]->bounds[2] = { midpoints[2], parentCell->bounds[2].y };
	// +X+Y-Z
	for (int i = 0; i < 2; ++i) parentCell->children[6]->bounds[i] = { midpoints[i], parentCell->bounds[i].y };
	parentCell->children[6]->bounds[2] = { parentCell->bounds[2].x, midpoints[2] };
	// +X+Y+Z
	for (int i = 0; i < 3; ++i) parentCell->children[7]->bounds[i] = { midpoints[i], parentCell->bounds[i].y };

	// Loop through each child cell and perform SAT with triangles
	for (int i = 0; i < 8; ++i) {
		OctreeCell* child = parentCell->children[i];
		for (auto it = parentCell->triangles.begin(); it != parentCell->triangles.end(); ++it) {
			const Triangle* triangle = world->getTriangle(*it);
			const cl_float3 vertices[3] = { world->getVertexBuffer()[triangle->face.x], world->getVertexBuffer()[triangle->face.y], world->getVertexBuffer()[triangle->face.z] };
			if (_mesh_triangleBoxIntersect(vertices, child->bounds)) {
				child->triangles.push_back(*it);
			}
		}

		// If is leaf node, add to leaf nodes
		if (child->depth == maxDepth && child->triangles.size() > 0) {
			leafnodes.push_back(child);
		}

		// Recursive octree children
		if (child->triangles.size() > 0 && child->depth < maxDepth) {
			_mesh_createOctreeChildren(child, world, maxDepth, leafnodes);
		}
	}
}

void Mesh::constructOctree(World* world, int depth, const cl_float2* bounds)
{
	octree.depth = 0;
	for (int i = 0; i < 3; ++i) octree.bounds[i] = bounds[i];
	_mesh_createOctreeChildren(&octree, world, depth, leafCells);

	if (depth == 0) {
		leafCells.push_back(&octree);
	}

	// Test to make sure every triangle is in the leaf cells
	for (auto tri = octree.triangles.begin(); tri != octree.triangles.end(); ++tri) {
		bool has = false;
		for (auto it = leafCells.begin(); it != leafCells.end(); ++it) {
			for (auto tit = (*it)->triangles.begin(); tit != (*it)->triangles.end(); ++tit) {
				if (*tit == *tri) {
					has = true;
					break;
				}
			}
			if (has) break;
		}
		if (!has) {
			std::cout << "Missing triangle: " << *tri << std::endl;
		}
	}
}
