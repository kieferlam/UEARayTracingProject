#define PI (3.14159265359f)
#define HPI (1.57079632679f)
#define SQ(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define SQRT33 (0.57735026919f)

#define TURN_FRACTION (1.61803398875f)

#define EPSILON (0.05f)

#define AIR_REFRACTIVE_INDEX (1.0f)
#define REFRACT_SURFACE_THICKNESS (1.0f)

#define MAX_VALUE (0xFFFFFFFF)

#define SPHERE_TYPE (0)
#define TRIANGLE_TYPE (1)

#define ROOT_TYPE (0)
#define REFLECT_TYPE (1)
#define REFRACT_TYPE (2)
#define SHADOW_TYPE (3)

#define NUM_SHADOW_RAYS (16)
#define SHADOW_RAY_DIST (0.1f)

#define BVH_PLANE_COUNT (7)

#define AMBIENT_STRENGTH (0.2f)
#define SPECULAR_STRENGTH (0.3f)

#define DAYLIGHT_COSINE_STRENGTH (0.7f)
#define DAYLIGHT_SHADOW_STRENGTH (0.7f)

#define MAX_RESULT_TREE_STACK (256)

#define print3f(v) printf("%f, %f, %f", v.x, v.y, v.z)

//#define SKIP_DDA

// types
typedef __constant unsigned char* SKYBOX;

typedef __constant unsigned int* TRIANGLE_GRID;

typedef __constant unsigned int* TRIANGLE_GRID_COUNT;

// Constants

__constant float3 daylight_direction = {SQRT33, -SQRT33, SQRT33};