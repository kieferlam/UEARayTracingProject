#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif


/**
This function just calculates the intersections of a ray and the scene/world.
Image processing and ray-combination should happen elsewhere.
*/
void trace(
    __constant RayConfig* input, 
    WorldPack pack,
    const Ray* ray, 
    __global TraceResult* result){

    result->hasTraced = true;
    // Sphere intersection
    float closest_T = MAX_VALUE;
    float closest_T2 = 0;
    int closest_i = -1;
    int closest_type = -1;
    for(int i = 0; i < pack.world->numSpheres; ++i){
        __constant Sphere* sphere = &pack.spheres[i];

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

    float planeDotRayOrigin[BVH_PLANE_COUNT];
    float planeDotRayDirection[BVH_PLANE_COUNT];
    // pre-compute ray with plane dot products
    for(int plane_i = 0; plane_i < BVH_PLANE_COUNT; ++plane_i){
        planeDotRayOrigin[plane_i] = dot(ray->origin, BVH_PlaneNormals[plane_i]);
        planeDotRayDirection[plane_i] = dot(ray->direction, BVH_PlaneNormals[plane_i]);
    }

    // Model intersections
    char closest_model = -1;
    float closest_model_T = MAX_VALUE;
    char closest_plane = -1;
    for(int i = 0; i < pack.world->numModels; ++i){
        __constant Model* model = pack.models + i;

        float tnear = -MAX_VALUE;
        float tfar = MAX_VALUE;
        uint planeIndex = -1;
        
        if(bvh_plane_intersect(model, planeDotRayOrigin, planeDotRayDirection, &tnear, &tfar, &planeIndex)){
            if(tnear < closest_model_T){
                closest_model_T = tnear;
                closest_plane = planeIndex;
                closest_model = i;
            }
        }
    }

    //If model bounding volume intersect
    if(closest_model_T < closest_T){
        float model_T = closest_model_T;
        int tri_i;
        if(model_intersect(pack, ray, closest_model, &model_T, &tri_i)){
            closest_T = model_T;
            closest_T2 = closest_T;
            closest_i = tri_i;
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
        result->normal = sphere_normal(&pack.spheres[result->objectIndex], result->intersect);
        result->material = pack.spheres[result->objectIndex].material;
    }else if(result->objectType == TRIANGLE_TYPE){
        result->normal = pack.triangles[closest_i].normal;
        result->material = pack.triangles[result->objectIndex].materialIndex;
    }
    result->cosine = fabs(dot(ray->direction, result->normal));

}

void trace_refract_exit(
    __constant RayConfig* config, 
    WorldPack pack,
    TraceResult* entry_result, 
    Ray* ray, 
    TraceResult* result){
    result->hasTraced = true;

    // Find intersection
    // Determine type
    if(entry_result->objectType == SPHERE_TYPE){
        SphereIntersect intersect_result;
        result->hasIntersect = sphere_intersect(ray, &pack.spheres[entry_result->objectIndex], &intersect_result);

        result->T = intersect_result.minT;
        result->T2 = intersect_result.maxT;
        result->intersect = ray->origin + ray->direction * result->T;
    }else if(entry_result->objectType == TRIANGLE_TYPE){
        result->hasIntersect = triangle_intersect(ray, &pack.triangles[entry_result->objectIndex], pack.vertices, &result->intersect, &result->T);
        result->T2 = result->T;
    }
    result->objectIndex = entry_result->objectIndex;

    if(!result->hasIntersect) return;

    if(entry_result->objectType == SPHERE_TYPE){
        result->normal = sphere_normal(&pack.spheres[result->objectIndex], result->intersect);
        result->material = pack.spheres[result->objectIndex].material;
    }else if(entry_result->objectType == TRIANGLE_TYPE){
        result->normal = pack.triangles[result->objectIndex].normal;
        result->material = pack.triangles[result->objectIndex].materialIndex;
    }

    result->cosine = dot(ray->direction, result->normal);
}

__kernel void RARTrace(
    __constant RayConfig* config, 
    __constant World* world, 
    __global TraceResult* results, 
    __constant float3* vertices, 
    __constant Material* materials,
    __constant Sphere* spheres,
    __constant Triangle* triangles,
    __constant Model* models,
    TRIANGLE_GRID triangleGrid,
    TRIANGLE_GRID_COUNT triangleCountGrid
    ){

    WorldPack pack = {world, vertices, materials, spheres, triangles, models, triangleGrid, triangleCountGrid};

    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int offset = (idx + (int)(idy * config->width)) * rar_getNumRays(config->bounces);
    __global TraceResult* baseResult = results + offset;
    
    generateEyeRay(&baseResult->ray, config, idx, idy);

    // Queue for processing new rays
    int queueTail = 0;
    int offsets[MAX_RESULT_TREE_STACK];
    offsets[queueTail] = 0;

    for(int i = 0; i <= queueTail && queueTail < MAX_RESULT_TREE_STACK - 3; ++i){
        int rayOffset = offsets[i];
        __global TraceResult* result = baseResult + rayOffset;
        Ray r = result->ray;
        trace(config, pack, &r, result);

        if(rar_getBounceNumber(offsets[i]) >= config->bounces) continue;
        
        // If intersect, add more rays
        if(result->hasIntersect){

            TraceResult localResult = *result;
            __constant Material* material = materials + localResult.material;

            // Add reflective ray
            if(material->reflectivity > EPSILON){
                // Queue reflection ray
                queueTail++;
                offsets[queueTail] = rar_getReflectChild(offsets[i]);
                // Create ray
                baseResult[offsets[queueTail]].ray.origin = localResult.intersect;
                getReflectDirection(&baseResult[offsets[queueTail]].ray.direction, r.direction, localResult.normal);
            }
            
            // Add refractive ray
            if(material->opacity < 1.0f - EPSILON){
                if(result->objectType == SPHERE_TYPE){ // Only trace exit ray if not triangle
                    // Find exit ray
                    Ray internal_ray; // This is the ray which will be traced inside the transparent object
                    internal_ray.origin = localResult.intersect;
                    local_getRefractDirection(&internal_ray.direction, r.direction, localResult.normal, AIR_REFRACTIVE_INDEX, material->refractiveIndex);
                    TraceResult refract_exit_result;
                    trace_refract_exit(config, pack, &localResult, &internal_ray, &refract_exit_result);
                    // Queue refraction ray
                    queueTail++;
                    offsets[queueTail] = rar_getRefractChild(offsets[i]);
                    if(refract_exit_result.hasIntersect){
                        __constant Material* refractMaterial = materials + refract_exit_result.material;
                        baseResult[offsets[queueTail]].ray.origin = refract_exit_result.intersect;
                        getRefractDirection(&baseResult[offsets[queueTail]].ray.direction, internal_ray.direction, -refract_exit_result.normal, refractMaterial->refractiveIndex, AIR_REFRACTIVE_INDEX);
                    }else{
                        baseResult[offsets[queueTail]].ray.origin = localResult.intersect; // Slightly refract the ray on infinitely small thickness
                        getRefractDirection(&baseResult[offsets[queueTail]].ray.direction, r.direction, -localResult.normal, 1.0f, AIR_REFRACTIVE_INDEX);
                    }
                }else{
                    baseResult[offsets[queueTail]].ray.origin = localResult.intersect; // Slightly refract the ray on infinitely small thickness
                    getRefractDirection(&baseResult[offsets[queueTail]].ray.direction, r.direction, -localResult.normal, 1.0f, AIR_REFRACTIVE_INDEX);
                }
            }

            // Add shadow ray
            if(material->opacity > EPSILON){
                // Queue the ray
                queueTail++;
                offsets[queueTail] = rar_getShadowChild(offsets[i]);
                // Create shadow ray
                baseResult[offsets[queueTail]].ray.origin = localResult.intersect;
                baseResult[offsets[queueTail]].ray.direction = -daylight_direction; // Shadow ray should be cast towards light source
            }
        }
    }
}
