#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

__kernel void ResetRays(__constant RayConfig* config, __global TraceResult* results){
    int idx = get_global_id(0);
    results[idx].hasTraced = false;
}
