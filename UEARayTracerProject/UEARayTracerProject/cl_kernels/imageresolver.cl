#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

typedef struct RayTraceNode RayTraceNode;
struct __attribute__ ((aligned(16))) RayTraceNode{
    __constant TraceResult* result;
    float3 reflectOutput;
    float3 refractOutput;
    float3 output;
    RayTraceNode* parent;
    RayTraceNode* reflectChild;
    RayTraceNode* refractChild;
    RayTraceNode* pad1;
    uint offset;
    uint visit;
    uint type; // 0 = root; 1 = reflection; 2 = refraction; 3 = shadow;
    uint processed;
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

RayTraceNode* addStack(RayTraceNode* stack, int* stackHead, __constant TraceResult* result, uint offset, uint type){
    if(*stackHead >= MAX_RESULT_TREE_STACK) return 0;
    RayTraceNode* next = stack + *stackHead;
    next->result = result;
    next->reflectChild = 0;
    next->refractChild = 0;
    next->reflectOutput = (float3)(0.0f, 0.0f, 0.0f);
    next->refractOutput = (float3)(0.0f, 0.0f, 0.0f);
    next->output = (float3)(0.0f, 0.0f, 0.0f);
    next->offset = offset;
    next->visit = 0;
    next->type = type;
    (*stackHead)++;
    return next;
}

__kernel void ResolveImage(__write_only image2d_t image, __constant RayConfig* config, __constant ImageConfig* imageConfig, __constant TraceResult* results, SKYBOX skybox, __constant Material* materials){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    
    int numRays = rar_getNumRays(config->bounces);
    int baseIndex = (idx + idy * config->width) * numRays;
    __constant TraceResult* baseResult = results + baseIndex;

    float3 final = (float3)(0.0f, 0.0f, 0.0f);

    if(!baseResult->hasIntersect){
        final = skybox_cubemap(imageConfig, skybox, baseResult->ray.direction);
    }else{
        // Stack for ray bounce processing
        RayTraceNode treeStack[MAX_RESULT_TREE_STACK];
        int stackHead = 0;
        
        // Add ray 
        addStack(treeStack, &stackHead, baseResult, 0, 0);

        // Add rays and their bounces to stack
        while(stackHead > 0 && stackHead < MAX_RESULT_TREE_STACK - 3){
            /**
            Conditions:
            Visit 1: Add reflect node
            Visit 2: Add refract node
            Visit 3: Add shadow node
            Visit 4: Process current node
            */

            RayTraceNode* currentNode = treeStack + stackHead - 1;
            currentNode->visit++;

            // These branches queue the children rays for processing
            // Child rays need to be processed first before calculating the current rays colour output
            if(currentNode->visit == REFLECT_TYPE){
                uint reflectChildIndex = rar_getReflectChild(currentNode->offset);
                if(baseResult[reflectChildIndex].hasTraced){ // Reflect child node
                    // Add reflect trace to stack
                    currentNode->reflectChild = addStack(treeStack, &stackHead, baseResult + reflectChildIndex, reflectChildIndex, REFLECT_TYPE);
                }
            }else if(currentNode->visit == REFRACT_TYPE){
                uint refractChildIndex = rar_getRefractChild(currentNode->offset);
                if(baseResult[refractChildIndex].hasTraced){
                    // Add refract trace to stack
                    currentNode->refractChild = addStack(treeStack, &stackHead, baseResult + refractChildIndex, refractChildIndex, REFRACT_TYPE);
                }
            }else{ 
                // This branch is where the the actual colour processing occurs
                
                float3 sky = skybox_cubemap(imageConfig, skybox, currentNode->result->ray.direction);

                if(currentNode->result->hasIntersect){
                    __constant Material* objectMaterial = materials + currentNode->result->material;
                    float kr = fresnel(currentNode->result->ray.direction, currentNode->result->normal, AIR_REFRACTIVE_INDEX, objectMaterial->refractiveIndex);
                    float daylight_cosine = 1.0f - max(-dot(currentNode->result->normal, daylight_direction), 0.0f) * DAYLIGHT_COSINE_STRENGTH;

                    // Calculate emission
                    float3 emission = objectMaterial->diffuse * daylight_cosine;
                   if(currentNode->refractChild->processed) emission = mix(currentNode->refractChild->output, emission, objectMaterial->opacity);

                    // Calculate reflection
                    float3 reflection = emission;
                    if(currentNode->reflectChild->processed) reflection = currentNode->reflectChild->output * daylight_cosine;

                    currentNode->output = mix(reflection, emission, kr);
                }else{
                    currentNode->output = sky;
                }

                currentNode->processed = true;
                stackHead--;
            }
        }

        final = treeStack->output;
        if(debug_isCenterPixel()) final = (float3)(1.0f, 0.0f, 0.0f);
    }

    // Write colour
    int2 coord = {idx, imageConfig->res.y - idy - 1};
    float4 colour = {final.x, final.y, final.z, 1.0f};
    write_imagef(image, coord, colour);
}
