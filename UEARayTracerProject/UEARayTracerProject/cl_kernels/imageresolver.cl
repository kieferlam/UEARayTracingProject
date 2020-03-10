#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

typedef struct __attribute__ ((aligned(16))){
    TraceResult* result;
    uint offset;
    uint visit;
}RayTraceNode;

float3 skybox_cubemap(__constant ImageResolverConfig* config, SKYBOX skybox_data, float3 dir){

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

__kernel void ResolveImage(__write_only image2d_t image, __constant RayConfig* rayConfig, __constant ImageResolverConfig* config, __constant TraceResult* results, SKYBOX skybox){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    
    int rayindex = (idx + idy * rayConfig->width) * rayConfig->bounces;

    int numRays = rar_getNumRays(rayConfig->bounces);

    float3 final = {0.0f, 0.0f, 0.0f};

    if(!results[rayindex].hasIntersect){
        final = skybox_cubemap(config, skybox, results[rayindex].ray.direction);
    }else{
        
        // Copy local results
        TraceResult localResults[64];
        for(int i = 0; i < numRays; ++i){
            localResults[i] = results[rayindex + i];
        }

        // Stack for ray bounce processing
        RayTraceNode treeStack[64];
        int stackHead = 0;
        
        // Add ray 
        treeStack[stackHead].result = localResults;
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
                    if(localResults[reflectChildIndex].hasTraced){
                        // Add reflect trace to stack
                        treeStack[stackHead].result = localResults + reflectChildIndex;
                        treeStack[stackHead].offset = reflectChildIndex;
                        treeStack[stackHead++].visit = 0;
                    }
                }
            }else if(currentNode->visit == 1){
                // int refractChildIndex = rar_getRefractChild(currentNode->offset);
                // if(refractChildIndex < numRays){
                //     if(localResults[refractChildIndex].hasTraced){
                //         // Add refract trace to stack
                //         treeStack[stackHead].result = localResults + refractChildIndex;
                //         treeStack[stackHead].offset = refractChildIndex;
                //         treeStack[stackHead++].visit = 0;
                //     }
                // }
            }else{
                if(currentNode->result->hasIntersect){
                    final = mix(final, currentNode->result->material.diffuse, 0.5f);
                }else{
                    final = mix(final, skybox_cubemap(config, skybox, currentNode->result->ray.direction), 0.5f);
                }
                stackHead--;
            }

            currentNode->visit++;
        }

    }

    // Write colour
    int2 coord = {idx, config->res.y - idy - 1};
    float4 colour = {final.x, final.y, final.z, 1.0f};
    write_imagef(image, coord, colour);
}
