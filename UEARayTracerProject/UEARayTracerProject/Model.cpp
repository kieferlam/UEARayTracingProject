#include "Model.h"
#include "OBJ_Loader.h"
#include <algorithm>
#include <limits>

Model::Model()
{
}

Model::~Model()
{
}

void Model::loadFromFile(const char* filename, World* world, float scale)
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
		int vertex_index_offset = world->getVertexBuffer().size();
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

			world->setTriangleMaterial(triangle, 0);
		}

		m->createBoundingVolume(world->getTriangle(0), world->getVertexBuffer());

		// Find min max bounds
		for (int i = 0; i < sizeof(mStruct.bounds) / sizeof(mStruct.bounds[0]); ++i) {
			mStruct.bounds[i].x = std::min(mStruct.bounds[i].x, m->getBounds(i).x);
			mStruct.bounds[i].y = std::max(mStruct.bounds[i].y, m->getBounds(i).y);
		}
	}

	mStruct.numTriangles = world->getTriangleCount() - mStruct.triangleOffset;

	modelStruct = world->addModel(mStruct);

}
