
#define SQRT33 (0.57735026919f)
__constant float3 BVH_PlaneNormals[] = {
	{1.0f, 0.0f, 0.0f},
	{0.0f, 1.0f, 0.0f},
	{0.0f, 0.0f, 1.0f},
	{SQRT33, SQRT33, SQRT33},
	{-SQRT33, SQRT33, SQRT33},
	{-SQRT33, -SQRT33, SQRT33},
	{SQRT33, -SQRT33, SQRT33},
};

typedef struct __attribute__ ((aligned(16))){
    float3 origin;
    float3 direction;
    uint2 pixelCoord;
} Ray;

typedef struct __attribute__ ((aligned(16))){
	float aspect;
	float width;
	float height;
	float screenDistance;
	float3 camera;
    float pitch;
    float yaw;
    uchar bounces;
} RayConfig;

typedef struct __attribute__ ((aligned(16))){
    float3 diffuse;
    float reflectivity;
    float opacity;
    float refractiveIndex;
} Material;

typedef struct __attribute__ ((aligned(16))){
    float3 position;
    float radius;
    uchar material;
} Sphere;

typedef struct __attribute__ ((aligned(16))) {
    float3 normal;
	int3 vertices;
	uchar materialIndex;
} Triangle;

typedef struct __attribute__ ((aligned(16))){
    float2 bounds[7];
    uint triangleOffset;
    uint numTriangles;
} Model;

typedef struct __attribute__ ((aligned(16))) {
	Sphere spheres[MAX_SPHERES];
	Triangle triangles[MAX_TRIANGLES];
    Model models[MAX_MODELS];
	uint numSpheres;
    uint numTriangles;
    uint numModels;
} World;

typedef struct TraceResult TraceResult;
struct __attribute__ ((aligned(16))) TraceResult{
    Ray ray;
    float3 intersect;
    float3 normal;
    float T;
    float T2;
    float cosine;
    uchar objectType;
    uchar objectIndex;
    uchar material;
    uchar bounce;
    bool hasIntersect;
    bool hasTraced;
};

typedef struct {
    float minT;
    float maxT;
} SphereIntersect;

typedef struct __attribute__ ((aligned(16))){
    int2 skyboxSize;
    int2 res;
} ImageConfig;