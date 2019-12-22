#define PI (3.14159265359f)
#define SQ(x) ((x)*(x))

#define MAX_SPHERES (8)
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

__kernel void TracerMain(__write_only image2d_t image, __constant KernelInput* input){
    int idx = get_global_id(0);
    int idy = get_global_id(1);

    float nx = 2.0f * (((float)(idx) / input->width) - 0.5f) * input->aspect;
    float ny = 2.0f * (((float)(idy) / input->height) - 0.5f);

    // Generate primary ray
    Ray primaryRay;
    primaryRay.origin = (float3)(nx, ny, input->screenDistance);
    primaryRay.direction = normalize(primaryRay.origin - input->camera);

    float4 final = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
    float exposure = 1.0f;

    float3 raw = trace_raw(input, &primaryRay);

    final = (float4)(raw, 0.0f) * exposure;
    
    // Write colour values
    int2 coord = {idx, input->height - idy};
    write_imagef(image, coord, final);
}