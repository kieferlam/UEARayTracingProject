#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif


typedef struct __attribute__ ((aligned(16))){
    TraceResult* result;
    TraceResult* parent;
    uint offset;
    uint visit;
    uint type; // 0 = -; 1 = reflection; 2 = refraction
}RayTraceNode;

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
    IMAGE FUNCTIONS
 */
float3 skybox_cubemap(__constant ImageConfig* config, SKYBOX skybox_data, float3 dir){

    const int skybox_img_size = config->skyboxSize.x * config->skyboxSize.y * 3;
    int face = 0;
    const int2 coord_indices[6] = {{2, 1}, {2, 1}, {0, 2}, {0, 2}, {0, 1}, {0, 1}};

    float3 absDir = (float3)(fabs(dir.x), fabs(dir.y), fabs(dir.z));
    bool polarity[3] = {dir.x > 0 ? true : false, dir.y > 0 ? true : false, dir.z > 0 ? true : false};

    float maxAxis, uc, vc;
    float2 uv;

    // POSITIVE X
    if(polarity[0] && absDir.x >= absDir.y && absDir.x >= absDir.z){
        // uv.u goes from +z to -z
        // uv.v goes from -y to +y
        maxAxis = absDir.x;
        uc = -dir.z;
        vc = dir.y;
        face = 0;
    }
    
    // NEGATIVE X
    if(!polarity[0] && absDir.x >= absDir.y && absDir.x >= absDir.z){
        // uv.u goes from +z to -z
        // uv.v goes from -y to +y
        maxAxis = absDir.x;
        uc = dir.z;
        vc = dir.y;
        face = 1;
    }
    
    // POSITIVE Y
    if(polarity[1] && absDir.y >= absDir.x && absDir.y >= absDir.z){
        // uv.u goes from -x to +x
        // uv.v goes from +z to -z
        maxAxis = absDir.y;
        uc = dir.x;
        vc = -dir.z;
        face = 2;
    }
    
    // NEGATIVE Y
    if(!polarity[1] && absDir.y >= absDir.x && absDir.y >= absDir.z){
        // uv.u goes from -x to +x
        // uv.v goes from -z to +z
        maxAxis = absDir.y;
        uc = dir.x;
        vc = dir.z;
        face = 3;
    }
    
    // POSITIVE Z
    if(polarity[2] && absDir.z >= absDir.x && absDir.z >= absDir.y){
        // uv.u goes from +x to -x
        // uv.v goes from -y to +y
        maxAxis = absDir.z;
        uc = -dir.x;
        vc = dir.y;
        face = 5;
    }
    
    // NEGATIVE Z
    if(!polarity[2] && absDir.z >= absDir.x && absDir.z >= absDir.y){
        // uv.u goes from -x to +x
        // uv.v goes from -y to +y
        maxAxis = absDir.z;
        uc = dir.x;
        vc = dir.y;
        face = 4;
    }
    
    uv = (float2)(0.5f * (uc / maxAxis + 1.0f), 0.5f * (vc / maxAxis + 1.0f));

    // Get offset in array corresponding to face
    int data_offset = skybox_img_size * face;

    // Get data offsets
    int x = (int)(uv.x * config->skyboxSize.x);
    int y = (int)(uv.y * config->skyboxSize.y);

    int pixelOffset = (y * config->skyboxSize.x + x) * 3;

    float r = skybox_data[data_offset + pixelOffset + 0] / 255.0f;
    float g = skybox_data[data_offset + pixelOffset + 1] / 255.0f;
    float b = skybox_data[data_offset + pixelOffset + 2] / 255.0f;
    
    return (float3)(r, g, b);
}

/**
    RAY TRACE
 */

/**
This function just calculates the intersections of a ray and the scene/world.
Image processing and ray-combination should happen elsewhere.
*/
void trace(__constant RayConfig* input, __constant World* world, Ray* ray, TraceResult* result){
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

__kernel void RayTrace(__write_only image2d_t image, __constant RayConfig* config, __constant ImageConfig* imageConfig, __constant World* world, SKYBOX skybox){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int numRays = rar_getNumRays(config->bounces);

    TraceResult results[64];
    
    generateEyeRay(&results[0].ray, config, idx, idy);

    // Queue for processing new rays
    int currentIndex = 0;
    int offsets[64];
    offsets[currentIndex] = 0;

    for(int i = 0; i <= currentIndex; ++i){
        int rayOffset = offsets[i];
        TraceResult* result = results + rayOffset;
        Ray r = result->ray;
        trace(config, world, &r, result);

        if(rar_getBounceNumber(offsets[i]) >= config->bounces) continue;
        
        // If intersect, add more rays
        if(result->hasIntersect){
            // TODO: Add checks if it even needs to create more rays
            // E.g. if the object is solid, no need for refract
            //      if the object is not reflective, no need for reflection

            // Queue reflection ray
            currentIndex++;
            offsets[currentIndex] = rar_getReflectChild(offsets[i]);
            // Create ray
            results[offsets[currentIndex]].ray.origin = result->intersect;
            getReflectDirection(&results[offsets[currentIndex]].ray.direction, r.direction, result->normal);

            // Find exit ray
            Ray internal_ray; // This is the ray which will be traced inside the transparent object
            internal_ray.origin = result->intersect;
            getRefractDirection(&internal_ray.direction, r.direction, result->normal, AIR_REFRACTIVE_INDEX, result->material.refractiveIndex);
            TraceResult refract_exit_result;
            trace_refract_exit(config, world, result, &internal_ray, &refract_exit_result);
            // Queue refraction ray
            currentIndex++;
            offsets[currentIndex] = rar_getRefractChild(offsets[i]);
            if(refract_exit_result.hasIntersect){
                results[offsets[currentIndex]].ray.origin = refract_exit_result.intersect;
                getRefractDirection(&results[offsets[currentIndex]].ray.direction, internal_ray.direction, -refract_exit_result.normal, refract_exit_result.material.refractiveIndex, AIR_REFRACTIVE_INDEX);
            }else{
                results[offsets[currentIndex]].ray.origin = result->intersect;
                getRefractDirection(&results[offsets[currentIndex]].ray.direction, r.direction, result->normal, AIR_REFRACTIVE_INDEX, result->material.refractiveIndex);
            }
        }

    }

    // Process colour
    float3 final = {0.0f, 0.0f, 0.0f};

    if(!results[0].hasIntersect){
        final = skybox_cubemap(imageConfig, skybox, results[0].ray.direction);
    }else{

        // Stack for ray bounce processing
        RayTraceNode treeStack[64];
        int stackHead = 0;
        
        // Add ray 
        treeStack[stackHead].result = results;
        treeStack[stackHead].offset = 0;
        treeStack[stackHead].visit = 0;

        // Add rays and their bounces to stack
        while(stackHead >= 0){
            /**
            Conditions:
            Visit 0: Add reflect node
            Visit 1: Add refract node
            Visit 2: Process Current Node
             */

            RayTraceNode* currentNode = treeStack + stackHead;

            if(currentNode->visit == 0){
                int reflectChildIndex = rar_getReflectChild(currentNode->offset);
                if(reflectChildIndex < numRays){
                    if(results[reflectChildIndex].hasTraced){ // Reflect child node
                        // Add reflect trace to stack
                        stackHead++;
                        RayTraceNode* reflectNode = treeStack + stackHead;
                        reflectNode->result = results + reflectChildIndex;
                        reflectNode->parent = currentNode->result;
                        reflectNode->offset = reflectChildIndex;
                        reflectNode->visit = 0;
                        reflectNode->type = 1;
                    }
                }
            }else if(currentNode->visit == 1){
                int refractChildIndex = rar_getRefractChild(currentNode->offset);
                if(refractChildIndex < numRays){
                    if(results[refractChildIndex].hasTraced){
                        // Add refract trace to stack
                        stackHead++;
                        RayTraceNode* refractNode = treeStack + stackHead;
                        refractNode->result = results + refractChildIndex;
                        refractNode->parent = currentNode->result;
                        refractNode->offset = refractChildIndex;
                        refractNode->visit = 0;
                        refractNode->type = 2;
                    }
                }
            }else{
                if(currentNode->result->hasIntersect){
                    final = mix(final, currentNode->result->material.diffuse, 0.5f);
                }else{
                    float mix_factor = 0.5f;
                    float cosine = fabs(currentNode->parent->cosine);
                    if(currentNode->type != 0) mix_factor = currentNode->type == 1 ? (1.0f - cosine) : (cosine);
                    final = mix(final, skybox_cubemap(imageConfig, skybox, currentNode->result->ray.direction), cosine);
                }
                stackHead--;
            }

            currentNode->visit++;
        }

    }

    // Write colour
    int2 coord = {idx, imageConfig->res.y - idy - 1};
    float4 colour = {final.x, final.y, final.z, 1.0f};
    write_imagef(image, coord, colour);
}
