#include "Model.h"
#include "OBJ_Loader.h"
#include <algorithm>
#include <limits>
#include <iomanip>

Model::Model()
{
}

Model::~Model()
{
}

void Model::loadFromFile(const char* filename, World* world, float scale, int mat)
{
	if (world == nullptr) {
		std::cout << "World ptr cannot be null. Could not load file." << std::endl;
		return;
	}

	objl::Loader loader;

	bool loaded = loader.LoadFile(filename);

	if (!loaded) {
		std::cout << "Could not load OBJ." << std::endl;
	}

	ModelStruct mStruct;

	// Reset bounds
	for (int i = 0; i < sizeof(mStruct.bounds) / sizeof(mStruct.bounds[0]); ++i) {
		mStruct.bounds[i].x = std::numeric_limits<float>::max();
		mStruct.bounds[i].y = std::numeric_limits<float>::min();
	}

	mStruct.triangleOffset = world->getTriangleCount();

	// Create Mesh objects
	for (auto mesh_it = loader.LoadedMeshes.begin(); mesh_it != loader.LoadedMeshes.end(); ++mesh_it) {
		meshes.push_back(Mesh());
		Mesh* m = &meshes[0];
		m->name = mesh_it->MeshName;

		/**
			Add vertices to world. The indices for the mesh need to be offset by the vertices already in the world object.
		*/
		size_t vertex_index_offset = world->getVertexBuffer().size();
		for (auto vertex_it = mesh_it->Vertices.begin(); vertex_it != mesh_it->Vertices.end(); ++vertex_it) {
			world->addVertex({ vertex_it->Position.X * scale, vertex_it->Position.Y * scale, vertex_it->Position.Z * scale});
		}

		for (int i = 0; i < mesh_it->Indices.size(); i += 3) {
			cl_uint3 face = { mesh_it->Indices[i] + vertex_index_offset, mesh_it->Indices[i+1] + vertex_index_offset, mesh_it->Indices[i+2] + vertex_index_offset };
			// Assuming the face only has 1 normal
			// The object contains normals for each vertex so the normal of a point on the face can be interpolated
			//cl_float3 normal = { mesh_it->Vertices[i].Normal.X, mesh_it->Vertices[i].Normal.Y, mesh_it->Vertices[i].Normal.Z };
			int triangle = world->addTriangle(face.x, face.y, face.z);
			m->addTriangle(triangle);

			world->setTriangleMaterial(triangle, mat);
		}

		std::cout << "Mesh " << m->name << " with " << m->getTriangleCount() << " triangles." << std::endl;

		// Create bounds for the mesh
		m->createBoundingVolume(world->getTriangle(0), world->getVertexBuffer());

		// Find min max bounds of the entire model
		for (int i = 0; i < sizeof(mStruct.bounds) / sizeof(mStruct.bounds[0]); ++i) {
			mStruct.bounds[i].x = std::min(mStruct.bounds[i].x, m->getBounds(i).x);
			mStruct.bounds[i].y = std::max(mStruct.bounds[i].y, m->getBounds(i).y);
		}
	}

	for (auto mesh_it = meshes.begin(); mesh_it != meshes.end(); ++mesh_it) {
		Mesh* m = &(*mesh_it);

		// Compute the size of the grid bounds
		cl_float boundsSize[3] = { mStruct.bounds[0].y - mStruct.bounds[0].x, mStruct.bounds[1].y - mStruct.bounds[1].x, mStruct.bounds[2].y - mStruct.bounds[2].x };
		cl_float cellSize[3];
		for (int i = 0; i < 3; ++i) cellSize[i] = boundsSize[i] / GRID_CELL_ROW_COUNT; // Calculate width, height, and length/depth of the grid cells

		std::cout << "Constructing triangle grid for mesh " << m->name << std::endl;

		world->addTriangleGrid(&mStruct.triangleGridOffset, &mStruct.triangleCountOffset);

		std::cout << "Constructing octree for mesh " << m->name << std::endl;
		m->constructOctree(world, GRID_CELL_DEPTH, mStruct.bounds);

		std::cout << "Adding octree leaf nodes to mesh grid" << std::endl;
		// Get leaf nodes and put into grid
		for (auto leaf = m->getLeafNodes().begin(); leaf != m->getLeafNodes().end(); ++leaf) {
			const OctreeCell* cell = *leaf;
			cl_float3 cellMid = { (cell->bounds[0].y + cell->bounds[0].x) * 0.5f, (cell->bounds[1].y + cell->bounds[1].x) * 0.5f, (cell->bounds[2].y + cell->bounds[2].x) * 0.5f };
			cl_float3 boundsOffset = { cellMid.x - mStruct.bounds[0].x, cellMid.y - mStruct.bounds[1].x, cellMid.z - mStruct.bounds[2].x };

			cl_int3 index = { (boundsOffset.x / boundsSize[0]) * GRID_CELL_ROW_COUNT, (boundsOffset.y / boundsSize[1]) * GRID_CELL_ROW_COUNT, (boundsOffset.z / boundsSize[2]) * GRID_CELL_ROW_COUNT };

			unsigned int coord = getGridOffset(index);

			for (auto leaf_tri = cell->triangles.begin(); leaf_tri != cell->triangles.end() && world->getTriangleCountGrid()[coord] < GRID_MAX_TRIANGLES_PER_CELL; ++leaf_tri) {
				world->addTriangleToGrid(*leaf_tri, coord);
				if (world->getTriangleCountGrid()[coord] == GRID_MAX_TRIANGLES_PER_CELL) {
					std::cout << "Max triangle count per cell reached." << std::endl;
				}
			}

		}
	}

	mStruct.numTriangles = world->getTriangleCount() - mStruct.triangleOffset;

	modelStruct = world->addModel(mStruct);

}
