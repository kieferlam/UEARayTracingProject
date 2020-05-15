#ifndef INCLUDES
#define INCLUDES
#include "defines.h"
#include "structs.h"
#endif

bool debug_isCenterPixel(){
    return get_global_id(0) == 1280 / 2 && get_global_id(1) == 720 / 2;
}

void swap(float* a, float* b){
    float temp = *a;
    *a = *b;
    *b = temp;
}

void generateEyeRay(__global Ray* output, __constant RayConfig* config, int x, int y){
    // Normalised coordinates
    float nx = 2.0f * (((float)(x) / config->width) - 0.5f) * config->aspect;
    float ny = 2.0f * (((float)(y) / config->height) - 0.5f);
    float nz = config->screenDistance;

    float3 coord = {nx, ny * cos(config->pitch) + nz * -sin(config->pitch), nz * cos(config->pitch) + ny * sin(config->pitch)};
    coord = (float3)(coord.x * cos(config->yaw) + coord.z * sin(config->yaw), coord.y, coord.z * cos(config->yaw) + coord.x * -sin(config->yaw));

    output->origin = coord + config->camera;
    output->direction = normalize(output->origin - config->camera);
}

float logbase(float base, float num){
    return log(num) / log(base);
}

int rar_getNumRays(int bounces){
    return (int)((1.0f-pow(NUM_RAY_CHILDREN, (float)bounces+1)) / (1.0f-NUM_RAY_CHILDREN));
}

int rar_getBounce(int index){
    return (int)ceil(logbase(NUM_RAY_CHILDREN, 1 + (NUM_RAY_CHILDREN-1)*index) - 1);
}

int rar_getParentIndex(int index){
    return (index-1) / NUM_RAY_CHILDREN;
}

int rar_getReflectChild(int index){
    return NUM_RAY_CHILDREN*index + REFLECT_TYPE;
}

int rar_getRefractChild(int index){
    return NUM_RAY_CHILDREN*index + REFRACT_TYPE;
}

int rar_getShadowChild(int index){
    return NUM_RAY_CHILDREN*index + SHADOW_TYPE;
}

void getReflectDirection(__global float3* direction_out, float3 direction_in, float3 normal){
    *direction_out = direction_in - 2.0f*dot(direction_in, normal)*normal;
}

void getRefractDirection(__global float3* direction_out, float3 direction_in, float3 normal, float from_index, float to_index){
    float n = from_index / to_index;
    float cosI = -dot(normal, direction_in);
    float sinT2 = n * n * (1.0f - cosI*cosI);
    if(sinT2 > 1.0f){
        *direction_out = direction_in;
        return;
    }
    float cosT = sqrt(1.0f - sinT2);
    *direction_out = n * direction_in + (n * cosI - cosT) * normal;
}

void local_getRefractDirection(float3* direction_out, float3 direction_in, float3 normal, float from_index, float to_index){
    float n = from_index / to_index;
    float cosI = -dot(normal, direction_in);
    float sinT2 = n * n * (1.0f - cosI*cosI);
    if(sinT2 > 1.0f){
        *direction_out = direction_in;
        return;
    }
    float cosT = sqrt(1.0f - sinT2);
    *direction_out = n * direction_in + (n * cosI - cosT) * normal;
}

float project(const float3 planeNormal, const float3 point){
    return dot(planeNormal, point);
}

float3 dda_getStep(const float3 cellSize, const float3 direction, const float3 stepPolarity){
    float3 invdir = 1.0f / direction;
    if(direction.x == 0.0f) invdir.x = 0.0f;
    if(direction.y == 0.0f) invdir.y = 0.0f;
    if(direction.z == 0.0f) invdir.z = 0.0f;
    return stepPolarity * cellSize * invdir;
}

float3 dda_getInitialT(const Ray* ray, const float boundT, const float3 gridmin, const float3 cellSize){
    float3 Ogrid = ray->origin + ray->direction * boundT - gridmin;
    float3 Ocell = Ogrid / cellSize;
    return (floor(Ocell) * cellSize - Ogrid) / ray->direction;
}

int3 dda_getCellOrigin(const float3 startPos, const float3 gridmin, const float3 cellSize){
    float3 cellO = (startPos - gridmin) / cellSize;
    return (int3)(
        (int) clamp(floor(cellO.x), 0.0f, (float)GRID_CELL_ROW_COUNT-1),
        (int) clamp(floor(cellO.y), 0.0f, (float)GRID_CELL_ROW_COUNT-1),
        (int) clamp(floor(cellO.z), 0.0f, (float)GRID_CELL_ROW_COUNT-1)
    );
}

float3 dda_getCellStepPolarity(const float3 direction){
    return (float3)(direction.x < 0 ? -1.0f : 1.0f, direction.y < 0 ? -1.0f : 1.0f, direction.z < 0 ? -1.0f : 1.0f);
}

uint getTriangleGridOffset(int3 cellindex){
    return cellindex.x * SQ(GRID_CELL_ROW_COUNT) + cellindex.y * GRID_CELL_ROW_COUNT + cellindex.z;
}

/**
    INTERSECT FUNCTIONS
 */

float3 sphere_normal(__constant Sphere* sphere, float3 surface){
    return normalize(surface - sphere->position);
}

bool sphere_intersect(const Ray* ray, __constant Sphere* sphere, SphereIntersect* result){
    float3 vec_raysphere = ray->origin - sphere->position; // This line should be omitted in the future for performance optimizations

    // at^2 + bt + c = 0
    float a = 1.0f; // dot(ray->direction, ray->direction) Since ray direction is normalized, this is just 1.0
    float b = 2.0f * dot(ray->direction, vec_raysphere);
    float c = dot(vec_raysphere, vec_raysphere) - SQ(sphere->radius);

    // t = (-b +- sqrt(b^2 - 4ac)) / 2a;
    float discriminant = SQ(b) - 4*a*c;
    if(discriminant < 0) return false;

    float dsqrt = sqrt(discriminant);
    result->minT = (-b - dsqrt) / 2.0f;
    result->maxT = (-b + dsqrt) / 2.0f;
    if(result->minT > result->maxT){
        swap(&result->minT, &result->maxT);
    }

    // Epsilon (Make sure ray from a sphere doesn't intersect itself)
    if(result->minT < EPSILON){
        result->minT = result->maxT;
    }

    // If intersect is behind origin, it doesn't intersect;
    if(result->maxT < EPSILON) return false;

    return true;
}

bool triangle_intersect(const Ray* ray, __constant Triangle* const_triangle, __constant float3* vertices, float3* intersect, float* T){
    // Copy to local/generic memory for faster operations
    Triangle triangle = *const_triangle;

    float3 edge1 = vertices[triangle.vertices[1]] - vertices[triangle.vertices[0]];
    float3 edge2 = vertices[triangle.vertices[2]] - vertices[triangle.vertices[0]];
    float3 h = cross(ray->direction, edge2);
    float a = dot(edge1, h);

    if(a > -EPSILON && a < EPSILON){
        return false; // Ray is parallel to triangle
    }

    float f = 1.0f / a;
    float3 s = ray->origin - vertices[triangle.vertices[0]];
    float u = f * dot(s, h);
    if(u < 0.0f || u > 1.0f){
        return false;
    }

    float3 q = cross(s, edge1);
    float v = f * dot(ray->direction, q);
    if(v < 0.0f || u + v > 1.0f){
        return false;
    }

    // Compute t
    float t = f * dot(edge2, q);
    if(t < EPSILON){
        return false;
    }

    *T = t;
    *intersect = ray->origin + ray->direction * t;

    return true;
}

int getCellBoundaryOffset(float a){
    return a < 0.0f ? 0 : 1;
}

bool model_intersect(
    WorldPack pack,
    const Ray* ray, 
    uchar modelIndex, 
    float* closest_T, 
    int* closest_I){
    
    float T = *closest_T;
    const Ray r = *ray;

    __constant Model* model = pack.models + modelIndex;

    // Step through model grid
    float3 gridmin = {model->bounds[0].x, model->bounds[1].x, model->bounds[2].x};
    float3 gridmax = {model->bounds[0].y, model->bounds[1].y, model->bounds[2].y};

    float3 cellSize = (gridmax - gridmin) / GRID_CELL_ROW_COUNT;
    float3 rayStart = (r.origin + r.direction * T * (1.0f + EPSILON));
    float3 dda_origin = rayStart - gridmin;

    float3 step = {
        r.direction.x < 0.0f ? -1.0f : 1.0f,
        r.direction.y < 0.0f ? -1.0f : 1.0f,
        r.direction.z < 0.0f ? -1.0f : 1.0f
    };
    if(r.direction.x == 0.0f) step.x = 0.0f;
    if(r.direction.y == 0.0f) step.y = 0.0f;
    if(r.direction.z == 0.0f) step.z = 0.0f;

    float3 invdir = 1.0f / r.direction;

    float3 deltaT = fabs(cellSize * invdir);

    int3 currentV = convert_int3(floor(dda_origin / cellSize));
    int3 cellBoundaryOffset = {r.direction.x > 0.0f, r.direction.y > 0.0f, r.direction.z > 0.0f};
    float3 cellBoundary = convert_float3(currentV + cellBoundaryOffset) * cellSize + gridmin;
    float3 Tv = (cellBoundary - r.origin) / r.direction;
    int max_step = GRID_CELL_ROW_COUNT * 2;

    bool hasIntersect = false;
    float closest_triangle_T = MAX_VALUE;
    while(
        currentV.x >= 0 && currentV.x < GRID_CELL_ROW_COUNT &&
        currentV.y >= 0 && currentV.y < GRID_CELL_ROW_COUNT &&
        currentV.z >= 0 && currentV.z < GRID_CELL_ROW_COUNT &&
        max_step > 0
    ){
        max_step--;
        uint celloffset = getTriangleGridOffset(currentV);

        for(int i = 0; i < pack.triangleCountGrid[model->triangleCountOffset + celloffset]; ++i){
            __constant Triangle* triangle = pack.triangles + pack.grid[model->triangleGridOffset + (celloffset * GRID_MAX_TRIANGLES_PER_CELL) + i];

            float3 intersect;
            float T;

            if(!triangle_intersect(ray, triangle, pack.vertices, &intersect, &T)) continue;

            if(T < closest_triangle_T){
                closest_triangle_T = T;
                *closest_T = T;
                *closest_I = i;
            }

            hasIntersect = true;
        }

        float3 incr = {
            (Tv.x <= Tv.y) && (Tv.x <= Tv.z),
            (Tv.y <= Tv.x) && (Tv.y <= Tv.z),
            (Tv.z <= Tv.x) && (Tv.z <= Tv.y),
        };


        Tv += incr * deltaT;
        currentV += convert_int3(incr * step);
    }
    return hasIntersect;
}

bool bvh_plane_intersect(__constant Model* model, float* planeDotOrigin, float* planeDotDirection, float* tNear, float* tFar, uint* planeIndex){
    for(uint plane_i = 0; plane_i < BVH_PLANE_COUNT; ++plane_i){
        float3 planeNormal = BVH_PlaneNormals[plane_i];

        float tNearPlane = (model->bounds[plane_i].x - planeDotOrigin[plane_i]) / planeDotDirection[plane_i];
        float tFarPlane = (model->bounds[plane_i].y - planeDotOrigin[plane_i]) / planeDotDirection[plane_i];
        if(planeDotDirection[plane_i] < 0.0f){
            swap(&tNearPlane, &tFarPlane);
        }
        
        if(tNearPlane > *tNear) *tNear = tNearPlane, *planeIndex = plane_i;
        if(tFarPlane < *tFar) *tFar = tFarPlane;
        if(*tNear > *tFar) return false;
    }
    return true;
}

/**
    RAY TRACE
 */

float3 reflect(float3 in, float3 normal){
    return in - 2.0f * dot(in, normal) * normal;
}

float fresnel(float3 in, float3 normal, float fromIOR, float toIOR){
    float cosI = dot(in, normal);
    float etaI = fromIOR;
    float etaT = toIOR;
    if(cosI > 0.0f){
        swap(&etaI, &etaT);
    }
    float n = etaI / etaT;
    float sinT2 = SQ(n) * max(0.0f, 1.0f - SQ(cosI));
    if(sinT2 > 1.0f) return 1.0f; // Total internal reflection
    float cosT = sqrt(1.0f - sinT2);
    cosI = fabs(cosI);
    float Rs = ((etaT * cosI) - (etaI * cosT)) / ((etaT * cosI) + (etaI * cosT));
    float Rp = ((etaI * cosI) - (etaT * cosT)) / ((etaI * cosI) + (etaT * cosT));
    return (SQ(Rs) + SQ(Rp)) / 2.0f;
}

float3 refract(float3 in, float3 normal, float fromIOR, float toIOR){
    float cosI = dot(in, normal);
    float etaI = fromIOR;
    float etaT = toIOR;
    float3 N = normal;
    if(cosI < 0.0f){
        cosI = -cosI;
    }else{
        swap(&etaI, &etaT);
        N = -N;
    }
    float n = etaI / etaT;
    float k = 1.0f - SQ(etaT) * (1.0f - SQ(cosI));
    return k < 0.0f ? (float3)(0.0f, 0.0f, 0.0f) : n * in + (n * cosI - sqrt(k)) * N;
}

float triangle_intersect_T(Ray* ray, Triangle* triangle, __constant float3* vertices){
    float d = dot(triangle->normal, vertices[triangle->vertices[0]]);
    float t = (dot(triangle->normal, ray->origin) + d) / dot(triangle->normal, ray->direction);
    return t;
}


/**
This function just calculates the intersections of a ray and the scene/world.
Image processing and ray-combination should happen elsewhere.
*/
void local_trace(
    __constant RayConfig* input, 
    WorldPack pack,
    const Ray* ray, 
    TraceResult* result){

    result->hasTraced = true;
    // Sphere intersection
    float closest_T = MAX_VALUE;
    float closest_T2 = 0;
    int closest_i = -1;
    int closest_type = -1;
    for(int i = 0; i < pack.world->numSpheres; ++i){
        __constant Sphere* sphere = &pack.spheres[i];

        float3 vec_raysphere = ray->origin - sphere->position;
        float dot_raysphere = dot(normalize(vec_raysphere), ray->direction);

        // If sphere is behind origin and origin is outside, skip
        if(-dot_raysphere < 0.0f && SQ(sphere->radius) < dot(vec_raysphere, vec_raysphere)) continue;

        // Find intersection
        SphereIntersect intersect_result;
        if(!sphere_intersect(ray, sphere, &intersect_result)) continue;

        // Check if closer
        if(intersect_result.minT < closest_T){
            closest_T = intersect_result.minT;
            closest_T2 = intersect_result.maxT;
            closest_i = i;
            closest_type = SPHERE_TYPE;
        }
    }

    float planeDotRayOrigin[BVH_PLANE_COUNT];
    float planeDotRayDirection[BVH_PLANE_COUNT];
    // pre-compute ray with plane dot products
    for(int plane_i = 0; plane_i < BVH_PLANE_COUNT; ++plane_i){
        planeDotRayOrigin[plane_i] = dot(ray->origin, BVH_PlaneNormals[plane_i]);
        planeDotRayDirection[plane_i] = dot(ray->direction, BVH_PlaneNormals[plane_i]);
    }

    // Model intersections
    char closest_model = -1;
    float closest_model_T = MAX_VALUE;
    char closest_plane = -1;
    for(int i = 0; i < pack.world->numModels; ++i){
        __constant Model* model = pack.models + i;

        float tnear = -MAX_VALUE;
        float tfar = MAX_VALUE;
        uint planeIndex = -1;
        
        if(bvh_plane_intersect(model, planeDotRayOrigin, planeDotRayDirection, &tnear, &tfar, &planeIndex)){
            if(tnear < closest_model_T){
                closest_model_T = tnear;
                closest_plane = planeIndex;
                closest_model = i;
            }
        }
    }

    // If model bounding volume intersect
    if(closest_model_T < closest_T){
        float model_T = closest_model_T;
        int tri_i;
        if(model_intersect(pack, ray, closest_model, &model_T, &tri_i)){
            closest_T = model_T;
            closest_T2 = closest_T;
            closest_i = tri_i;
            closest_type = TRIANGLE_TYPE;
        }
    }

    // for(int i = 0; i < pack.world->numTriangles; ++i){
    //     __constant Triangle* tri = pack.triangles + i; 
    //     float3 tri_intersect;
    //     float tri_T;
    //     if(triangle_intersect(ray, tri, pack.vertices, &tri_intersect, &tri_T)){
    //         if(tri_T < closest_T){
    //             closest_T = tri_T;
    //             closest_T2 = closest_T;
    //             closest_i = i;
    //             closest_type = TRIANGLE_TYPE;
    //         }
    //     }
    // }

    // If no intersect, stop
    if(closest_i < 0){
        result->hasIntersect = false;
        return;
    }

    result->hasIntersect = true;
    result->T = closest_T;
    result->T2 = closest_T2;
    result->intersect = ray->origin + ray->direction * result->T;
    result->objectIndex = closest_i;
    result->objectType = closest_type;

    // Find the normal of the sphere
    if(result->objectType == SPHERE_TYPE){
        result->normal = sphere_normal(&pack.spheres[result->objectIndex], result->intersect);
        result->material = pack.spheres[result->objectIndex].material;
    }else if(result->objectType == TRIANGLE_TYPE){
        result->normal = pack.triangles[closest_i].normal;
        result->material = pack.triangles[result->objectIndex].materialIndex;
    }
    result->cosine = fabs(dot(ray->direction, result->normal));

}

void trace(
    __constant RayConfig* input, 
    WorldPack pack,
    const Ray* ray, 
    __global TraceResult* result){
    TraceResult localResult;
    local_trace(input, pack, ray, &localResult);
    *result = localResult;
}

// Image resolve

float3 phong(TraceResult* trace, Material* material){
    float intensity = clamp(dot(trace->normal, -daylight_direction), AMBIENT_STRENGTH, 1.0f);
    float3 diffuse = intensity * material->diffuse;

    //  Blinn-Phong Half Vector
    float3 H = normalize(daylight_direction + trace->ray.direction);

    // specular intensity
    intensity = pow(clamp(dot(-trace->normal, H), 0.0f, 1.0f), material->specular) * SPECULAR_STRENGTH;
    float3 specular = (float3)(intensity, intensity, intensity);

    return diffuse + specular;
}

// float3 calc_emission(){

// }