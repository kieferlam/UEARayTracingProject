#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

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

/**
    RAY TRACE
 */

float3 reflect(float3 in, float3 normal){
    return in - 2.0f * dot(in, normal) * normal;
}

float3 refract(float3 incident, float3 normal, float n1, float n2){
    float n = n1 / n2;
    float cosI = -dot(normal, incident);
    float sinT2 = SQ(n) * (1.0f - SQ(cosI));
    if(sinT2 > 1.0f) return incident; // Total internal reflection
    float cosT = sqrt(1.0 - sinT2);
    return n * incident + (n * cosI - cosT) * normal;
}

/**
This function just calculates the intersections of a ray and the scene/world.
Image processing and ray-combination should happen elsewhere.
*/
void trace(__constant RayConfig* input, __constant World* world, Ray* ray, __global TraceResult* result){
    result->hasTraced = true;
    // Sphere intersection
    SphereIntersect closest_intersect = {MAX_VALUE, MAX_VALUE};
    int closest_i = -1;
    for(int i = 0; i < world->numSpheres; ++i){
        __constant Sphere* sphere = &world->spheres[i];

        float3 vec_raysphere = ray->origin - sphere->position;
        float dot_raysphere = dot(normalize(vec_raysphere), ray->direction);

        // If sphere is behind origin and origin is outside, skip
        if(-dot_raysphere < 0.0f && SQ(sphere->radius) < dot(vec_raysphere, vec_raysphere)) continue;

        // Find intersection
        SphereIntersect intersect_result;
        if(!sphere_intersect(ray, sphere, &intersect_result)) continue;

        // Check if closer
        if(intersect_result.minT < closest_intersect.minT){
            closest_intersect = intersect_result;
            closest_i = i;
        }
    }

    // If no intersect, stop
    if(closest_i < 0){
        result->hasIntersect = false;
        return;
    }

    result->hasIntersect = true;
    result->T = closest_intersect.minT;
    result->T2 = closest_intersect.maxT;
    result->intersect = ray->origin + ray->direction * result->T;
    result->sphereIndex = closest_i;

    // Find the normal of the sphere
    result->normal = sphere_normal(&world->spheres[result->sphereIndex], result->intersect);
    result->cosine = dot(ray->direction, result->normal);

    // Get material
    result->material = world->spheres[result->sphereIndex].material;

}

void trace_refract_exit(__constant RayConfig* config, __constant World* world, TraceResult* entry_result, Ray* ray, TraceResult* result){
    result->hasTraced = true;

    // Find intersection
    SphereIntersect intersect_result;
    result->hasIntersect = sphere_intersect(ray, &world->spheres[entry_result->sphereIndex], &intersect_result);

    result->T = intersect_result.minT;
    result->T2 = intersect_result.maxT;
    result->intersect = ray->origin + ray->direction * result->T;
    result->sphereIndex = entry_result->sphereIndex;

    result->normal = sphere_normal(&world->spheres[result->sphereIndex], result->intersect);
    result->cosine = dot(ray->direction, result->normal);

    result->material = world->spheres[result->sphereIndex].material;
}

__kernel void RARTrace(__constant RayConfig* config, __constant World* world, __global TraceResult* results){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int offset = (idx + (int)(idy * config->width)) * config->bounces;
    __global TraceResult* baseResult = results + offset;
    
    generateEyeRay(&results[offset].ray, config, idx, idy);

    // Queue for processing new rays
    int currentIndex = 0;
    int offsets[256];
    offsets[currentIndex] = 0;

    for(int i = 0; i <= currentIndex; ++i){
        int rayOffset = offsets[i];
        __global TraceResult* result = baseResult + rayOffset;
        Ray r = result->ray;
        trace(config, world, &r, result);

        if(rar_getBounceNumber(offsets[i]) >= config->bounces - 1) continue;
        
        // If intersect, add more rays
        if(result->hasIntersect){
            // TODO: Add checks if it even needs to create more rays
            // E.g. if the object is solid, no need for refract
            //      if the object is not reflective, no need for reflection

            TraceResult localResult = *result;

            // Queue reflection ray
            currentIndex++;
            offsets[currentIndex] = rar_getReflectChild(offsets[i]);
            // Create ray
            results[offsets[currentIndex] + offset].ray.origin = localResult.intersect;
            getReflectDirection(&results[offsets[currentIndex]].ray.direction, r.direction, localResult.normal);

            // Find exit ray
            // Ray internal_ray; // This is the ray which will be traced inside the transparent object
            // internal_ray.origin = localResult.intersect;
            // local_getRefractDirection(&internal_ray.direction, r.direction, localResult.normal, AIR_REFRACTIVE_INDEX, localResult.material.refractiveIndex);
            // TraceResult refract_exit_result;
            // trace_refract_exit(config, world, &localResult, &internal_ray, &refract_exit_result);
            // // Queue refraction ray
            // currentIndex++;
            // offsets[currentIndex] = rar_getRefractChild(offsets[i]);
            // if(refract_exit_result.hasIntersect){
            //     results[offsets[currentIndex] + offset].ray.origin = localResult.intersect;
            //     getRefractDirection(&results[offsets[currentIndex] + offset].ray.direction, internal_ray.direction, refract_exit_result.normal, refract_exit_result.material.refractiveIndex, AIR_REFRACTIVE_INDEX);
            // }else{
            //     results[offsets[currentIndex] + offset].ray.origin = refract_exit_result.intersect;
            //     getRefractDirection(&results[offsets[currentIndex] + offset].ray.direction, r.direction, localResult.normal, AIR_REFRACTIVE_INDEX, localResult.material.refractiveIndex);
            // }
        }

    }
}
