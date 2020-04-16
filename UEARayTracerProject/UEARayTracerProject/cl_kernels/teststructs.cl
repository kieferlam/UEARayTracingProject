#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

void setModelFields(__constant Model* in, __global Model* out){
    out->triangleGridOffset = in->triangleGridOffset;
    out->triangleCountOffset = in->triangleCountOffset;
    for(int i = 0; i < 7; ++i){
        out->bounds[i] = in->bounds[i];
    }
    out->triangleOffset = in->triangleOffset;
    out->numTriangles = in->numTriangles;
}

void setWorldFields(__constant World* in, __global World* out){
    out->spheres[MAX_SPHERES-1] = in->spheres[MAX_SPHERES-1];
    out->triangles[MAX_TRIANGLES-1] = in->triangles[MAX_TRIANGLES-1];
    out->models[MAX_MODELS-1] = in->models[MAX_MODELS-1];
    out->numSpheres = in->numSpheres;
    out->numTriangles = in->numTriangles;
    out->numModels = in->numModels;
}

__kernel void TestStructs(__constant Model* in_model, __global Model* out_model, __constant World* in_world, __global World* out_world){
    setModelFields(in_model, out_model);
    setWorldFields(in_world, out_world);
}