#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#endif

void swap(float* a, float* b){
    float temp = *a;
    *a = *b;
    *b = temp;
}

void generateEyeRay(__global Ray* output, __constant RayConfig* config, int x, int y){
    // Normalised coordinates
    float nx = 2.0f * (((float)(x) / config->width) - 0.5f) * config->aspect;
    float ny = 2.0f * (((float)(y) / config->height) - 0.5f);
    float nz = config->screenDistance;

    float3 coord = {nx, ny * cos(config->pitch) + nz * -sin(config->pitch), nz * cos(config->pitch) + ny * sin(config->pitch)};
    coord = (float3)(coord.x * cos(config->yaw) + coord.z * sin(config->yaw), coord.y, coord.z * cos(config->yaw) + coord.x * -sin(config->yaw));

    output->origin = coord + config->camera;
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

float project(float3 planeNormal, float3 point){
    return dot(planeNormal, point);
}

/**
    INTERSECT FUNCTIONS
 */

float3 sphere_normal(__constant Sphere* sphere, float3 surface){
    return normalize(surface - sphere->position);
}

bool sphere_intersect(Ray* ray, __constant Sphere* sphere, SphereIntersect* result){
    float3 vec_raysphere = ray->origin - sphere->position; // This line should be omitted in the future for performance optimizations

    // at^2 + bt + c = 0
    float a = 1.0f; // dot(ray->direction, ray->direction) Since ray direction is normalized, this is just 1.0
    float b = dot(ray->direction * vec_raysphere, (float3)(2.0f, 2.0f, 2.0f));
    float c = dot(SQ(vec_raysphere), (float3)(1.0f, 1.0f, 1.0f)) - SQ(sphere->radius);

    // t = (-b +- sqrt(b^2 - 4ac)) / 2a;
    float discriminant = SQ(b) - 4*a*c;
    if(discriminant < 0) return false;

    result->minT = (-b - sqrt(SQ(b) - 4.0f*a*c)) / 2.0f;
    result->maxT = (-b + sqrt(SQ(b) - 4.0f*a*c)) / 2.0f;
    if(result->minT > result->maxT){
        float temp = result->minT;
        result->minT = result->maxT;
        result->maxT = temp;
    }

    // Epsilon (Make sure ray from a sphere doesn't intersect itself)
    if(result->minT < EPSILON){
        result->minT = result->maxT;
    }

    // If intersect is behind origin, it doesn't intersect;
    if(result->maxT < EPSILON) return false;

    return true;
}

bool triangle_intersect(Ray* ray, __constant Triangle* const_triangle, __constant float3* vertices, float3* intersect, float* T){
    // Copy to local/generic memory for faster operations
    Triangle triangle = *const_triangle;

    float3 edge1 = vertices[triangle.vertices[1]] - vertices[triangle.vertices[0]];
    float3 edge2 = vertices[triangle.vertices[2]] - vertices[triangle.vertices[0]];
    float3 h = cross(ray->direction, edge2);
    float a = dot(edge1, h);

    if(a > -EPSILON && a < EPSILON){
        return false; // Ray is parallel to triangle
    }

    float f = 1.0f / a;
    float3 s = ray->origin - vertices[triangle.vertices[0]];
    float u = f * dot(s, h);
    if(u < 0.0f || u > 1.0f){
        return false;
    }

    float3 q = cross(s, edge1);
    float v = f * dot(ray->direction, q);
    if(v < 0.0f || u + v > 1.0f){
        return false;
    }

    // Compute t
    float t = f * dot(edge2, q);
    if(t < EPSILON){
        return false;
    }

    *T = t;
    *intersect = ray->origin + ray->direction * t;

    return true;
}

bool bvh_plane_intersect(__constant Model* model, float* planeDotOrigin, float* planeDotDirection, float* tNear, float* tFar, uint* planeIndex){
    for(int plane_i = 0; plane_i < BVH_PLANE_COUNT; ++plane_i){
        float3 planeNormal = BVH_PlaneNormals[plane_i];

        float tNearPlane = (model->bounds[plane_i].x - planeDotOrigin[plane_i]) / planeDotDirection[plane_i];
        float tFarPlane = (model->bounds[plane_i].y - planeDotOrigin[plane_i]) / planeDotDirection[plane_i];
        if(planeDotDirection[plane_i] < 0.0f){
            swap(&tNearPlane, &tFarPlane);
        }

        if(tNearPlane > *tNear) *tNear = tNearPlane, planeIndex = plane_i;
        if(tFarPlane < *tFar) *tFar = tFarPlane;
        if(*tNear > *tFar) return false;
    }
    return true;
}

/**
    RAY TRACE
 */

float3 reflect(float3 in, float3 normal){
    return in - 2.0f * dot(in, normal) * normal;
}

float fresnel(float3 in, float3 normal, float fromIOR, float toIOR){
    float cosI = dot(in, normal);
    float etaI = fromIOR;
    float etaT = toIOR;
    if(cosI > 0.0f){
        swap(&etaI, &etaT);
    }
    float n = etaI / etaT;
    float sinT2 = SQ(n) * max(0.0f, 1.0f - SQ(cosI));
    if(sinT2 > 1.0f) return 1.0f; // Total internal reflection
    float cosT = sqrt(1.0f - sinT2);
    cosI = fabs(cosI);
    float Rs = ((etaT * cosI) - (etaI * cosT)) / ((etaT * cosI) + (etaI * cosT));
    float Rp = ((etaI * cosI) - (etaT * cosT)) / ((etaI * cosI) + (etaT * cosT));
    return (SQ(Rs) + SQ(Rp)) / 2.0f;
}

float3 refract(float3 in, float3 normal, float fromIOR, float toIOR){
    float cosI = dot(in, normal);
    float etaI = fromIOR;
    float etaT = toIOR;
    float3 N = normal;
    if(cosI < 0.0f){
        cosI = -cosI;
    }else{
        swap(&etaI, &etaT);
        N = -N;
    }
    float n = etaI / etaT;
    float k = 1.0f - SQ(etaT) * (1.0f - SQ(cosI));
    return k < 0.0f ? N : n * in + (n * cosI - sqrt(k)) * N;
}

float triangle_intersect_T(Ray* ray, Triangle* triangle, __constant float3* vertices){
    float d = dot(triangle->normal, vertices[triangle->vertices[0]]);
    float t = (dot(triangle->normal, ray->origin) + d) / dot(triangle->normal, ray->direction);
    return t;
}