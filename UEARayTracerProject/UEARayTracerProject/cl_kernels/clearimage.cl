#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#include "func.h"
#endif

__kernel void ClearImage(__write_only image2d_t image, __constant ImageConfig* imageConfig){
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    
    // Write colour
    int2 coord = {idx, imageConfig->res.y - idy - 1};
    float4 colour = {0.0f, 0.0f, 0.0f, 1.0f};
    write_imagef(image, coord, colour);
}
