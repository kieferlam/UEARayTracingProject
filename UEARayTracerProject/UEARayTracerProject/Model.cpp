#include "Model.h"
#include "OBJ_Loader.h"

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

	// Create Mesh objects
	for (auto mesh_it = loader.LoadedMeshes.begin(); mesh_it != loader.LoadedMeshes.end(); ++mesh_it) {
		Mesh m;
		m.name = mesh_it->MeshName;

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
			m.addTriangle(triangle);

			world->setTriangleMaterial(triangle, 2);
		}
	}

}
