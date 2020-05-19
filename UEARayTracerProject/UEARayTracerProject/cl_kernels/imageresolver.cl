#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

typedef struct RayTraceNode RayTraceNode;
struct __attribute__ ((aligned(16))) RayTraceNode{
    int reflectIndex;
    int refractIndex;
    int reflectChild;
    int refractChild;
    uint index;
    uint visit;
    uint type; // 0 = root; 1 = reflection; 2 = refraction; 3 = shadow;
    uint processed;
    int pad1;
};

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

    float3 col = {r, g, b};
    
    return col;
}

int pushStack(RayTraceNode* stack, int* stackHead, uint index, uint type){
    int readhead = *stackHead;
    if(readhead >= MAX_RESULT_TREE_STACK) return 0;
    RayTraceNode* next = stack + readhead;
    next->reflectIndex = -1;
    next->refractIndex = -1;
    next->reflectChild = 0;
    next->refractChild = 0;
    next->index = index;
    next->visit = 0;
    next->type = type;
    next->processed = false;
    (*stackHead)++;
    return readhead;
}

__kernel void ResolveImage(__write_only image2d_t image, __constant RayConfig* config, __constant ImageConfig* imageConfig, __global TraceResult* results, SKYBOX skybox, __constant Material* materials){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    
    int numRays = rar_getNumRays(config->bounces);
    int baseIndex = (idx + idy * config->width) * numRays;
    __global TraceResult* baseResult = results + baseIndex;

    float3 final = (float3)(0.0f, 0.0f, 0.0f);

    if(!baseResult->hasIntersect){
        final = skybox_cubemap(imageConfig, skybox, baseResult->ray.direction);
    }else{
        // Stack for ray bounce processing
        RayTraceNode treeStack[MAX_RESULT_TREE_STACK];
        float3 outputStack[MAX_RESULT_TREE_STACK];
        int stackHead = 0;
        int iterHead = 0;

        // Copy results to local memory
        __private TraceResult localResults[MAX_RESULT_TREE_STACK];
        for(int i = 0; i < numRays; ++i){
            localResults[i] = baseResult[i];
        }
        
        // Add ray 
        pushStack(treeStack, &stackHead, 0, 0);

        // Adding all rays to stack
        while(iterHead < stackHead && iterHead >= 0){
            /**
            Conditions:
            Visit 1: Add reflect node
            Visit 2: Add refract node
            Visit 4: Process current node
            */

            RayTraceNode* currentNode = treeStack + iterHead;
            currentNode->visit++;

            if(currentNode->visit == REFLECT_TYPE){
                uint reflectChildIndex = rar_getReflectChild(currentNode->index);
                if(localResults[reflectChildIndex].hasTraced){ // Reflect child node
                    // Add reflect trace to stack
                    currentNode->reflectChild = reflectChildIndex;
                    currentNode->reflectIndex = pushStack(treeStack, &stackHead, reflectChildIndex, REFLECT_TYPE);
                    iterHead++;
                }
            }else if(currentNode->visit == REFRACT_TYPE){
                uint refractChildIndex = rar_getRefractChild(currentNode->index);
                if(localResults[refractChildIndex].hasTraced){
                    // Add refract trace to stack
                    currentNode->refractChild = refractChildIndex;
                    currentNode->refractIndex = pushStack(treeStack, &stackHead, refractChildIndex, REFRACT_TYPE);
                    iterHead++;
                }
            }else{
                iterHead--;
            }
        }

        // Iterate through stack (popping, so iterating through the array backwards)
        while(stackHead >= 0){
            RayTraceNode* currentNode = treeStack + stackHead;

            TraceResult* result = localResults + currentNode->index;
            if(result->hasIntersect){
                Material objectMaterial = materials[result->material];
                float kr = fresnel(result->ray.direction, result->normal, AIR_REFRACTIVE_INDEX, objectMaterial.refractiveIndex);

                float daylight_cosine = 1.0f - max(dot(result->normal, daylight_direction), 0.0f) * DAYLIGHT_COSINE_STRENGTH;
                uint shadowChildIndex = rar_getShadowChild(currentNode->index);
                TraceResult* shadowResult = localResults + shadowChildIndex;

                // Calculate emission
                float3 transmission = phong(result, &objectMaterial);
                if(currentNode->refractIndex > -1) transmission = mix(outputStack[currentNode->refractChild], transmission, objectMaterial.opacity-EPSILON*2.0f);

                // Calculate reflection
                float3 reflection = transmission;
                if(currentNode->reflectIndex > -1) reflection = outputStack[currentNode->reflectChild];

                // Transform kr based on opacity
                kr = mix(kr, 1.0f - kr, objectMaterial.opacity);

                float3 out = transmission * (1.0f - kr) + reflection * kr;

                // Calculate shadows
                if(shadowResult->hasTraced && shadowResult->hasIntersect){
                    out *= 1.0f - (DAYLIGHT_SHADOW_STRENGTH * (1.0f - shadowResult->shadowSoftness) * materials[shadowResult->material].opacity);
                }

                outputStack[currentNode->index] = out;
            }else{
                float3 sky = skybox_cubemap(imageConfig, skybox, result->ray.direction);
                outputStack[currentNode->index] = sky;
            }

            stackHead--;
        }

        final = outputStack[0];
    }

    if(debug_isCenterPixel()){
        final = (float3)(1.0f, 0.0f, 0.0f);
    }

    // Write colour
    int2 coord = {idx, imageConfig->res.y - idy - 1};
    float4 colour = {final.x, final.y, final.z, 1.0f};
    write_imagef(image, coord, colour);
}
