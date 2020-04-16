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

int rar_getNumRays(int bounces){
    return (int)pow(2, (float)bounces + 1) - 1;
}

int rar_getParentIndex(int index){
    return (index-1) / 2;
}

int rar_getReflectChild(int index){
    return 2*index + 1;
}

int rar_getRefractChild(int index){
    return 2*index + 2;
}

int rar_getBounceNumber(int index){
    return (int)(floor(log2((float)index + 1)));
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

float3 dda_getStep(const float3 cellSize, const float3 direction){
    return fabs(cellSize / direction);
}

float3 dda_getInitialT(const float3 cellSize, const Ray* ray, const float3 gridmin){
    float3 Ogrid = ray->origin - gridmin;
    float3 Ocell = Ogrid / cellSize;
    return (floor(Ocell) * cellSize - Ogrid) / ray->direction;
}

int3 dda_getCellOrigin(const float3 rayOrigin, const float3 gridmin, const float3 cellSize){
    float3 cellO = (rayOrigin - gridmin) / cellSize;
    return (int3)(
        (int) cellO.x,
        (int) cellO.y,
        (int) cellO.z
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

bool sphere_intersect(Ray* ray, __constant Sphere* sphere, SphereIntersect* result){
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

// bool model_intersect(__constant World* world, __constant float3* vertices, const Ray* ray, uchar modelIndex, float* closest_T, int* closest_I){
//     __constant Model* model = world->models + modelIndex;

//     // Step through model grid
//     float3 gridmin = {model->bounds[0].x, model->bounds[1].x, model->bounds[2].x};
//     float3 gridmax = {model->bounds[0].y, model->bounds[1].y, model->bounds[2].y};
//     float3 griddim = gridmax - gridmin;
    
//     float3 cellSize = (griddim) / GRID_CELL_ROW_COUNT;
//     float3 deltaT, nextCrossingT;
//     float3 rayOGrid = ray->origin - gridmin;
//     float3 t;

//     if(ray->direction.x < 0){
//         deltaT.x = -GRID_CELL_ROW_COUNT / ray->direction.x;
//         t.x = (floor(rayOGrid.x / cellSize.x) * cellSize.x - rayOGrid.x) / ray->direction.x;
//     }else{
//         deltaT.x = GRID_CELL_ROW_COUNT / ray->direction.x;
//         t.x = ((floor(rayOGrid.x / cellSize.x) + 1) * cellSize.x - rayOGrid.x) / ray->direction.x;
//     }
//     if(ray->direction.y < 0){
//         deltaT.y = -GRID_CELL_ROW_COUNT / ray->direction.y;
//         t.y = (floor(rayOGrid.y / cellSize.y) * cellSize.y - rayOGrid.y) / ray->direction.y;
//     }else{
//         deltaT.y = GRID_CELL_ROW_COUNT / ray->direction.y;
//         t.y = ((floor(rayOGrid.y / cellSize.y) + 1) * cellSize.y - rayOGrid.y) / ray->direction.y;
//     }
//     if(ray->direction.z < 0){
//         deltaT.z = -GRID_CELL_ROW_COUNT / ray->direction.z;
//         t.z = (floor(rayOGrid.z / cellSize.z) * cellSize.z - rayOGrid.z) / ray->direction.z;
//     }else{
//         deltaT.z = GRID_CELL_ROW_COUNT / ray->direction.z;
//         t.z = ((floor(rayOGrid.z / cellSize.z) + 1) * cellSize.z - rayOGrid.z) / ray->direction.z;
//     }

//     float T = 0.0f;
//     float tNextCrossing = 0.0f;
//     int3 cellIndex = {rayOGrid.x / cellSize.x, rayOGrid.y / cellSize.y, rayOGrid.z / cellSize.z};

//     while(true){
//         if(t.x < t.y){
//             if(t.x < t.z){
//                 T = t.x;
//                 t.x += deltaT.x;
//                 if(ray->direction.x < 0) {
//                     cellIndex.x -= 1;
//                 }else{
//                     cellIndex.x += 1;
//                 }
//             }else{
//                 T = t.z;
//                 t.z += deltaT.z;
//                 if(ray->direction.z < 0) {
//                     cellIndex.z -= 1;
//                 }else{
//                     cellIndex.z += 1;
//                 }
//             }
//         }else{
//             if(t.y < t.z){
//                 T = t.y;
//                 t.y += deltaT.y;
//                 if(ray->direction.y < 0) {
//                     cellIndex.y -= 1;
//                 }else{
//                     cellIndex.y += 1;
//                 }
//             }else{
//                 T = t.z;
//                 t.z += deltaT.z;
//                 if(ray->direction.z < 0) {
//                     cellIndex.z -= 1;
//                 }else{
//                     cellIndex.z += 1;
//                 }
//             }
//         }

//         if(cellIndex.x < 0 || cellIndex.y < 0 || cellIndex.z < 0 ||
//         cellIndex.x > GRID_CELL_ROW_COUNT - 1 || cellIndex.y > GRID_CELL_ROW_COUNT - 1 || cellIndex.z > GRID_CELL_ROW_COUNT - 1){
//             break;
//         } 
//     }
// }

bool model_intersect(
    __constant World* world, 
    __constant float3* vertices, 
    TRIANGLE_GRID grid, 
    __constant unsigned int* triangleCountGrid, 
    const Ray* ray, 
    uchar modelIndex, 
    float* closest_T, 
    int* closest_I){

    __constant Model* model = world->models + modelIndex;

    // Step through model grid
    float3 gridmin = {model->bounds[0].x, model->bounds[1].x, model->bounds[2].x};
    float3 gridmax = {model->bounds[0].y, model->bounds[1].y, model->bounds[2].y};
    union{
        float3 vector;
        float array[3];
    } cellSize;
    cellSize.vector = (gridmax - gridmin) / GRID_CELL_ROW_COUNT;

    union{
        float array[4];
        float3 vector;
    } dtCell;
    dtCell.vector = dda_getStep(cellSize.vector, ray->direction);
    union{
        int array[3];
        int3 vector;
    } cellindex;
    cellindex.vector = dda_getCellOrigin(ray->origin, gridmin, cellSize.vector);

    union{
        float array[4];
        float3 vector;
    } stepPolarity;
    stepPolarity.vector = dda_getCellStepPolarity(ray->direction);
    
    union{
        float3 vector;
        float array[3];
    } cumT;
    cumT.vector = dda_getInitialT(cellSize.vector, ray, gridmin);

    while(
        ((stepPolarity.vector.x < 0.0f && cellindex.vector.x >= 0) || (stepPolarity.vector.x >= 0.0f && cellindex.vector.x < GRID_CELL_ROW_COUNT)) &&
        ((stepPolarity.vector.y < 0.0f && cellindex.vector.y >= 0) || (stepPolarity.vector.y >= 0.0f && cellindex.vector.y < GRID_CELL_ROW_COUNT)) &&
        ((stepPolarity.vector.z < 0.0f && cellindex.vector.z >= 0) || (stepPolarity.vector.z >= 0.0f && cellindex.vector.z < GRID_CELL_ROW_COUNT))
    ){

        // Intersect test with cell's triangles
        // Triangle intersections
        bool hasIntersect = false;
        uint celloffset = getTriangleGridOffset(cellindex.vector);
        float closest_triangle_T = MAX_VALUE;

        for(int i = 0; i < triangleCountGrid[model->triangleCountOffset + celloffset]; ++i){
            __constant Triangle* triangle = world->triangles + (grid[model->triangleGridOffset + celloffset + i]);

            float3 intersect;
            float T;

            if(!triangle_intersect(ray, triangle, vertices, &intersect, &T)) continue;

            if(T < closest_triangle_T){
                closest_triangle_T = T;
                *closest_T = T;
                *closest_I = i;
            }

            hasIntersect = true;
        }
        if(hasIntersect) return true;

        // Find lowest T value so we know which dimension to increment
        int lowI = 0;
        float lowT = cumT.array[0];
        for(int i = 1; i < 3; ++i){
            if(cumT.array[i] < lowT){
                lowT = cumT.array[i];
                lowI = i;
            }
        }

        cumT.array[lowI] += dtCell.array[lowI];

        cellindex.array[lowI] += stepPolarity.array[lowI] * 1.0f;
    }

    return false;
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
    return k < 0.0f ? N : n * in + (n * cosI - sqrt(k)) * N;
}

float triangle_intersect_T(Ray* ray, Triangle* triangle, __constant float3* vertices){
    float d = dot(triangle->normal, vertices[triangle->vertices[0]]);
    float t = (dot(triangle->normal, ray->origin) + d) / dot(triangle->normal, ray->direction);
    return t;
}