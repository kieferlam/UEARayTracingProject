#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

__kernel void ResetRays(__constant RayConfig* config, __global TraceResult* results){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int numRays = rar_getNumRays(config->bounces);
    
    int baseIndex = (idx + idy * config->width) * numRays;
    __global TraceResult* baseResult = results + baseIndex;

    for(int i = 0; i < numRays; ++i){
        baseResult[i].hasTraced = false;
    }
}
