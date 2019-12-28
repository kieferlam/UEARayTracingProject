#define PI (3.14159265359f)
#define SQ(x) ((x)*(x))

#define MAX_VALUE (0xFFFFFFFF)

__constant float3 FORWARD = (float3)(0.0f, 0.0f, 1.0f);

typedef struct{
    float3 origin;
    float3 direction;
} Ray;

typedef struct __attribute__ ((aligned(16))){
    float3 position;
    float3 colour;
    float radius;
} SphereStruct;

typedef struct __attribute__ ((aligned(16))){
    float aspect;
    float width;
    float height;
    float screenDistance;
    float3 camera;
    SphereStruct spheres[MAX_SPHERES];
    int numSpheres;
} KernelInput;

typedef struct __attribute__ ((aligned(16))){
    int2 skyboxSize;
    bool shadows;
    bool reflection;
    bool refraction;
} RTConfig;

typedef struct {
    bool hasIntersect;
    float T;
    float T2;
    float3 normal;
    int sphereIndex;
} TraceResult;

typedef struct {
    float minT;
    float maxT;
} SphereIntersect;

/**
    INTERSECT FUNCTIONS
 */

float3 sphere_normal(__constant SphereStruct* sphere, float3 surface){
    return normalize(surface - sphere->position);
}

bool sphere_intersect(Ray* ray, __constant SphereStruct* sphere, SphereIntersect* result){
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

    // If intersect is behind origin, it doesn't intersect;
    if(result->maxT < 0) return false;

    return true;
}

/**
    SKYBOX
 */

float3 skybox_cubemap(__constant RTConfig* config, __constant unsigned char* skybox_data, float3 dir){
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
        // uv.u goes from -x to +x
        // uv.v goes from -y to +y
        maxAxis = absDir.z;
        uc = dir.x;
        vc = dir.y;
        face = 4;
    }
    
    // NEGATIVE Z
    if(polarity[2] && absDir.z >= absDir.x && absDir.z >= absDir.y){
        // uv.u goes from +x to -x
        // uv.v goes from -y to +y
        maxAxis = absDir.z;
        uc = -dir.x;
        vc = dir.y;
        face = 5;
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

void trace_ray(__constant KernelInput* input, Ray* ray, TraceResult* result){
    SphereIntersect closest_intersect = {MAX_VALUE, MAX_VALUE};
    int closest_i = -1;
    for(int i = 0; i < input->numSpheres; ++i){
        __constant SphereStruct* sphere = &input->spheres[i];

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

    // Find the normal of the sphere
    result->normal = sphere_normal(&input->spheres[result->sphereIndex], ray->origin + ray->direction * result->T);

    result->sphereIndex = closest_i;
}

float3 trace_raw(__constant KernelInput* input, Ray* primaryRay){
    float3 values = (float3)(0.0f, 0.0f, 0.1f);

    TraceResult result;
    trace_ray(input, primaryRay, &result);

    if(result.hasIntersect){
        values += input->spheres[result.sphereIndex].colour;
    }

    return values;
}

__kernel void TracerMain(__write_only image2d_t image, __constant KernelInput* input, __constant RTConfig* config, __constant unsigned char* skybox){
    // These are the global IDs for the current instance of the kernel
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    // Normalised coordinates
    float nx = 2.0f * (((float)(idx) / input->width) - 0.5f) * input->aspect;
    float ny = 2.0f * (((float)(idy) / input->height) - 0.5f);

    // Generate primary ray
    Ray primaryRay;
    primaryRay.origin = (float3)(nx, ny, input->screenDistance);
    primaryRay.direction = normalize(primaryRay.origin - input->camera);

    float4 final = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    float exposure = 1.0f;

    float3 raw = trace_raw(input, &primaryRay);
    
    raw = skybox_cubemap(config, skybox, primaryRay.direction);

    final = (float4)(raw, 0.0f) * exposure;
    
    // Write colour values
    int2 coord = {idx, input->height - idy};
    write_imagef(image, coord, final);
}