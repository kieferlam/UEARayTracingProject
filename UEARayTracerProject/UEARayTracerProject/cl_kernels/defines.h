#define PI (3.14159265359f)
#define SQ(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define SQRT33 (0.57735026919f)

#define EPSILON (0.001f)

#define AIR_REFRACTIVE_INDEX (1.005f)
#define REFRACT_SURFACE_THICKNESS (1.0f)

#define MAX_VALUE (0xFFFFFFFF)

#define SPHERE_TYPE (0)
#define TRIANGLE_TYPE (1)

#define ROOT_TYPE (0)
#define REFLECT_TYPE (1)
#define REFRACT_TYPE (2)
#define SHADOW_TYPE (3)

#define BVH_PLANE_COUNT (7)

#define DAYLIGHT_COSINE_STRENGTH (0.7f)

#define MAX_RESULT_TREE_STACK (243)

// types
typedef __constant unsigned char* SKYBOX;

typedef __constant unsigned int* TRIANGLE_GRID;

typedef __constant unsigned int* TRIANGLE_GRID_COUNT;

// Constants

__constant float3 daylight_direction = {-SQRT33, SQRT33, -SQRT33};