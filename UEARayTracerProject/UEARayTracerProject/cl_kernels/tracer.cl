#define PI (3.14159265359f)
#define SQ(x) ((x)*(x))

#define MAX_SPHERES (64)

typedef struct{
    float3 position;
    float4 colour;
    float radius;
} SphereStruct;

typedef struct {
    SphereStruct spheres[MAX_SPHERES];
    int numSpheres;
} KernelInput;

__kernel void TracerMain(__write_only image2d_t image, __constant KernelInput* input){
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    int2 coord = {idx, idy};

    float4 col = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

    write_imagef(image, coord, col);
}