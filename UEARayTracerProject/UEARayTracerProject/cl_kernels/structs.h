
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
    uint bounces;
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
    uint material;
} Sphere;

typedef struct __attribute__ ((aligned(16))) {
    float3 normal;
	int3 vertices;
	uint materialIndex;
} Triangle;

typedef struct __attribute__ ((aligned(16))){
    float2 bounds[8]; // Only 7 axis but the 8th is for padding (struct alignment)
    uint triangleOffset;
    uint pad1[3];
    uint numTriangles;
    uint pad2[3];
	uint triangleGridOffset;
    uint pad3[3];
    uint triangleCountOffset;
} Model;

typedef struct __attribute__ ((aligned(16))) {
	uint numSpheres;
    uint pad1[3];
    uint numTriangles;
    uint pad2[3];
    uint numModels;
    uint pad3[3];
} World;

typedef struct TraceResult TraceResult;
struct __attribute__ ((aligned(16))) TraceResult{
    Ray ray;
    float3 intersect;
    float3 normal;
    float T;
    float T2;
    float cosine;
    float pad1;
    uint objectType;
    uint objectIndex;
    uint material;
    uint bounce;
    bool hasIntersect;
    bool hasTraced;
    bool pad2[14];
};

typedef struct __attribute__ ((aligned(16))) {
    __constant World* world;
    __constant float3* vertices;
    __constant Material* materials;
    __constant Sphere* spheres;
    __constant Triangle* triangles;
    __constant Model* models;
    TRIANGLE_GRID grid;
    TRIANGLE_GRID_COUNT triangleCountGrid;
} WorldPack;

typedef struct {
    float minT;
    float maxT;
} SphereIntersect;

typedef struct __attribute__ ((aligned(16))){
    int2 skyboxSize;
    int2 res;
} ImageConfig;