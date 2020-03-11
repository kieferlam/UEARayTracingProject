typedef struct __attribute__ ((aligned(16))){
    float3 origin;
    float3 direction;
    int2 pixelCoord;
} Ray;

typedef struct {
	float aspect;
	float width;
	float height;
	float screenDistance;
	float3 camera;
    int bounces;
} RayConfig;

typedef struct __attribute__ ((aligned(16))){
    float3 diffuse;
    float reflectivity;
    float opacity;
    float refractiveIndex;
} Material;

typedef struct __attribute__ ((aligned(16))){
    Material material;
    float3 position;
    float radius;
} Sphere;

typedef struct __attribute__ ((aligned(16))) {
    float3 normal;
	int3 vertices;
	int materialIndex;
} Triangle;

typedef struct __attribute__ ((aligned(16))) {
	Sphere spheres[MAX_SPHERES];
	Triangle triangles[MAX_TRIANGLES];
	int numSpheres;
    int numTriangles;
} World;

typedef struct TraceResult TraceResult;
struct TraceResult{
    Material material;
    Ray ray;
    float3 intersect;
    float3 normal;
    float T;
    float T2;
    float cosine;
    int sphereIndex;
    int bounce;
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