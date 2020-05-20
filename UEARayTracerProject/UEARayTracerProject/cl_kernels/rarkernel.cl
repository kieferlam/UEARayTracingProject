#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif


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
    baseResult->bounce = 0;
    baseResult->rayType = ROOT_TYPE;
    
    generateEyeRay(&baseResult->ray, config, idx, idy);

    // Queue for processing new rays
    int queueTail = 0;
    int offsets[MAX_RESULT_TREE_STACK];
    offsets[queueTail] = 0;

    for(int i = 0; i <= queueTail && queueTail < MAX_RESULT_TREE_STACK - NUM_RAY_CHILDREN; ++i){
        int rayOffset = offsets[i];
        __global TraceResult* result = baseResult + rayOffset;
        Ray r = result->ray;
        trace(config, &pack, &r, result);

        // If is shadow ray, cast multiple rays to find softness
        if(result->rayType == SHADOW_TYPE){
            Ray softShadowRay;
            softShadowRay.origin = r.origin;
		    float3 axis = fabs(r.direction.x) > 0.1f ? (float3)(0.0f, 1.0f, 0.0f) : (float3)(1.0f, 0.0f, 0.0f);
            float3 u = normalize(cross(axis, r.direction));
            float3 v = cross(r.direction, u);
            int numHit = result->hasIntersect;
            for(int i = 0; i < NUM_SHADOW_RAYS; ++i){
                float dist = pow(i / (NUM_SHADOW_RAYS - 1.0f), 0.5f) * SHADOW_RAY_DIST;
                float angle = 2.0f * PI * TURN_FRACTION * i;
                float sx = dist * cos(angle);
                float sy = dist * sin(angle);
                softShadowRay.direction = normalize(dist * sx * u + dist * sy * v + r.direction);

                TraceResult shadowResult;
                local_trace(config, &pack, &softShadowRay, &shadowResult);
                if(shadowResult.hasIntersect) numHit++;
            }

            result->shadowSoftness = 1.0f - pow((float)numHit / (float)(NUM_SHADOW_RAYS + 1), 8.0f);
        }

        if(result->bounce >= config->bounces) continue;

        // If intersect, add more rays
        if(result->hasIntersect && result->rayType != SHADOW_TYPE){

            TraceResult localResult = *result;
            __constant Material* material = materials + localResult.material;

            // Add reflective ray
            if(material->reflectivity > EPSILON){
                // Queue reflection ray
                queueTail++;
                offsets[queueTail] = rar_getReflectChild(rayOffset);
                // Create ray
                baseResult[offsets[queueTail]].ray.origin = localResult.intersect;
                getReflectDirection(&baseResult[offsets[queueTail]].ray.direction, r.direction, localResult.normal);
                baseResult[offsets[queueTail]].bounce = localResult.bounce + 1;
                baseResult[offsets[queueTail]].rayType = REFLECT_TYPE;
            }
            
            // Add refractive ray
            if(material->opacity < 1.0f - EPSILON){
                if(localResult.objectType == SPHERE_TYPE){ // Only trace exit ray if not triangle
                    __constant Sphere* sphere = spheres + localResult.objectIndex;
                    // Calculate internal ray direction
                    float3 internal_direction;
                    local_getRefractDirection(&internal_direction, r.direction, localResult.normal, AIR_REFRACTIVE_INDEX, material->refractiveIndex);
                    // Calculate exit ray origin
                    /**
                        This calculates the angle between the internal ray direction and the normal of the sphere.
                        Used to find the distance from the internal ray origin (original intersection point) to the exit ray origin.
                        Together with the internal ray direction, the exit ray origin can be calculated.
                    */
                    float internal_theta = fabs(dot(internal_direction, localResult.normal));
                    float internal_length = sin(internal_theta * HPI) * sphere->radius * 2.0f;
                    
                    // Get child ray
                    queueTail++;
                    offsets[queueTail] = rar_getRefractChild(rayOffset);

                    // Set exit ray origin
                    baseResult[offsets[queueTail]].ray.origin = localResult.intersect + internal_direction * internal_length;
                    
                    // Calculate exit ray direction
                    float3 exit_normal = normalize(baseResult[offsets[queueTail]].ray.origin - sphere->position);
                    getRefractDirection(&baseResult[offsets[queueTail]].ray.direction, internal_direction, -exit_normal, material->refractiveIndex, AIR_REFRACTIVE_INDEX);
                    baseResult[offsets[queueTail]].bounce = localResult.bounce + 1;
                    baseResult[offsets[queueTail]].rayType = REFRACT_TYPE;
                }else{
                    queueTail++;
                    offsets[queueTail] = rar_getRefractChild(rayOffset);
                    baseResult[offsets[queueTail]].ray.origin = localResult.intersect; // Slightly refract the ray on infinitely small thickness
                    float3 refractNormal = -localResult.normal;
                    if(dot(r.direction, refractNormal) > 0) refractNormal = -refractNormal;
                    getRefractDirection(&baseResult[offsets[queueTail]].ray.direction, r.direction, refractNormal, 1.0f, material->refractiveIndex);
                    baseResult[offsets[queueTail]].bounce = localResult.bounce + 1;
                    baseResult[offsets[queueTail]].rayType = REFRACT_TYPE;
                }
            }

            // Add shadow ray
            if(material->opacity > EPSILON){
                // Queue the ray
                queueTail++;
                offsets[queueTail] = rar_getShadowChild(rayOffset);
                // Create shadow ray
                baseResult[offsets[queueTail]].ray.origin = localResult.intersect;
                baseResult[offsets[queueTail]].ray.direction = -daylight_direction; // Shadow ray should be cast towards light source
                baseResult[offsets[queueTail]].bounce = localResult.bounce + 1;
                baseResult[offsets[queueTail]].rayType = SHADOW_TYPE;
            }
        }
    }
}
