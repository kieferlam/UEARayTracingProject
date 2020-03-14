#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#endif

void generateEyeRay(__global Ray* output, __constant RayConfig* config, int x, int y){
    // Normalised coordinates
    float nx = 2.0f * (((float)(x) / config->width) - 0.5f) * config->aspect;
    float ny = 2.0f * (((float)(y) / config->height) - 0.5f);

    output->origin = (float3)(nx, ny, config->screenDistance);
    output->direction = normalize(output->origin - config->camera);
}

int rar_getNumRays(int bounces){
    return (int)pow(2, (float)bounces + 1) - 1;
}

int rar_getParentIndex(int index){
    return (index-1) / 2;
}

int rar_getReflectChild(int index){
    return 2*index + 1;
}

int rar_getRefractChild(int index){
    return 2*index + 2;
}

int rar_getBounceNumber(int index){
    return (int)(floor(log2((float)index + 1)));
}

void getReflectDirection(__global float3* direction_out, float3 direction_in, float3 normal){
    *direction_out = direction_in - 2.0f*dot(direction_in, normal)*normal;
}

void getRefractDirection(__global float3* direction_out, float3 direction_in, float3 normal, float from_index, float to_index){
    float n = from_index / to_index;
    float cosI = -dot(normal, direction_in);
    float sinT2 = n * n * (1.0f - cosI*cosI);
    if(sinT2 > 1.0f){
        *direction_out = direction_in;
        return;
    }
    float cosT = sqrt(1.0f - sinT2);
    *direction_out = n * direction_in + (n * cosI - cosT) * normal;
}

void local_getRefractDirection(float3* direction_out, float3 direction_in, float3 normal, float from_index, float to_index){
    float n = from_index / to_index;
    float cosI = -dot(normal, direction_in);
    float sinT2 = n * n * (1.0f - cosI*cosI);
    if(sinT2 > 1.0f){
        *direction_out = direction_in;
        return;
    }
    float cosT = sqrt(1.0f - sinT2);
    *direction_out = n * direction_in + (n * cosI - cosT) * normal;
}

float triangle_intersect_T(Ray* ray, Triangle* triangle, __constant float3* vertices){
    float d = dot(triangle->normal, vertices[triangle->vertices[0]]);
    float t = -(dot(triangle->normal, ray->origin) + d) / dot(triangle->normal, ray->direction);
    return t;
}