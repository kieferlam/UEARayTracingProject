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
void trace(__constant RayConfig* input, __constant World* world, __constant float3* vertices, Ray* ray, __global TraceResult* result){
    result->hasTraced = true;
    // Sphere intersection
    float closest_T = MAX_VALUE;
    float closest_T2 = 0;
    int closest_i = -1;
    int closest_type = -1;
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
        if(intersect_result.minT < closest_T){
            closest_T = intersect_result.minT;
            closest_T2 = intersect_result.maxT;
            closest_i = i;
            closest_type = SPHERE_TYPE;
        }
    }

    // Triangle intersections
    int closest_triangle = -1;
    for(int i = 0; i < world->numTriangles; ++i){
        __constant Triangle* triangle = world->triangles + i;

        float3 intersect;
        float T;

        if(!triangle_intersect(ray, triangle, vertices, &intersect, &T)) continue;

        if(T < closest_T){
            closest_T = T;
            closest_T2 = T;
            closest_i = i;
            closest_type = TRIANGLE_TYPE;
        }
    }

    // If no intersect, stop
    if(closest_i < 0){
        result->hasIntersect = false;
        return;
    }

    result->hasIntersect = true;
    result->T = closest_T;
    result->T2 = closest_T2;
    result->intersect = ray->origin + ray->direction * result->T;
    result->objectIndex = closest_i;
    result->objectType = closest_type;

    // Find the normal of the sphere
    if(result->objectType == SPHERE_TYPE){
        result->normal = sphere_normal(&world->spheres[result->objectIndex], result->intersect);
        result->material = world->spheres[result->objectIndex].material;
    }else if(result->objectType == TRIANGLE_TYPE){
        result->normal = world->triangles[closest_i].normal;
        result->material = world->triangles[result->objectIndex].materialIndex;
    }
    result->cosine = SQ(dot(ray->direction, result->normal));

}

void trace_refract_exit(__constant RayConfig* config, __constant World* world, __constant float3* vertices, TraceResult* entry_result, Ray* ray, TraceResult* result){
    result->hasTraced = true;

    // Find intersection
    // Determine type
    if(entry_result->objectType == SPHERE_TYPE){
        SphereIntersect intersect_result;
        result->hasIntersect = sphere_intersect(ray, &world->spheres[entry_result->objectIndex], &intersect_result);

        result->T = intersect_result.minT;
        result->T2 = intersect_result.maxT;
        result->intersect = ray->origin + ray->direction * result->T;
    }else if(entry_result->objectType == TRIANGLE_TYPE){
        result->hasIntersect = triangle_intersect(ray, &world->triangles[entry_result->objectIndex], vertices, &result->intersect, &result->T);
        result->T2 = result->T;
    }
    result->objectIndex = entry_result->objectIndex;

    if(!result->hasIntersect) return;

    if(entry_result->objectType == SPHERE_TYPE){
        result->normal = sphere_normal(&world->spheres[result->objectIndex], result->intersect);
        result->material = world->spheres[result->objectIndex].material;
    }else if(entry_result->objectType == TRIANGLE_TYPE){
        result->normal = world->triangles[result->objectIndex].normal;
        result->material = world->triangles[result->objectIndex].materialIndex;
    }

    result->cosine = SQ(dot(ray->direction, result->normal));
}

__kernel void RARTrace(__constant RayConfig* config, __constant World* world, __global TraceResult* results, __constant float3* vertices, __constant Material* materials){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int offset = (idx + (int)(idy * config->width)) * rar_getNumRays(config->bounces);
    __global TraceResult* baseResult = results + offset;
    
    generateEyeRay(&baseResult->ray, config, idx, idy);

    // Queue for processing new rays
    int queueTail = 0;
    int offsets[256];
    offsets[queueTail] = 0;

    for(int i = 0; i <= queueTail; ++i){
        int rayOffset = offsets[i];
        __global TraceResult* result = baseResult + rayOffset;
        Ray r = result->ray;
        trace(config, world, vertices, &r, result);

        if(rar_getBounceNumber(offsets[i]) >= config->bounces) continue;
        
        // If intersect, add more rays
        if(result->hasIntersect){
            // TODO: Add checks if it even needs to create more rays
            // E.g. if the object is solid, no need for refract
            //      if the object is not reflective, no need for reflection

            TraceResult localResult = *result;
            __constant Material* material = materials + localResult.material;

            if(material->reflectivity > EPSILON){
                // Queue reflection ray
                queueTail++;
                offsets[queueTail] = rar_getReflectChild(offsets[i]);
                // // Create ray
                baseResult[offsets[queueTail]].ray.origin = localResult.intersect;
                getReflectDirection(&baseResult[offsets[queueTail]].ray.direction, r.direction, localResult.normal);
            }
            
            if(material->opacity < 1.0f - EPSILON){
                // Find exit ray
                Ray internal_ray; // This is the ray which will be traced inside the transparent object
                internal_ray.origin = localResult.intersect;
                local_getRefractDirection(&internal_ray.direction, r.direction, localResult.normal, AIR_REFRACTIVE_INDEX, material->refractiveIndex);
                TraceResult refract_exit_result;
                trace_refract_exit(config, world, vertices, &localResult, &internal_ray, &refract_exit_result);
                // Queue refraction ray
                queueTail++;
                offsets[queueTail] = rar_getRefractChild(offsets[i]);
                if(refract_exit_result.hasIntersect){
                    __constant Material* refractMaterial = materials + refract_exit_result.material;
                    baseResult[offsets[queueTail]].ray.origin = refract_exit_result.intersect;
                    getRefractDirection(&baseResult[offsets[queueTail]].ray.direction, internal_ray.direction, -refract_exit_result.normal, refractMaterial->refractiveIndex, AIR_REFRACTIVE_INDEX);
                }else{
                    baseResult[offsets[queueTail]].ray.origin = localResult.intersect;
                    baseResult[offsets[queueTail]].ray.direction = r.direction;
                }
            }
        }
    }
}
