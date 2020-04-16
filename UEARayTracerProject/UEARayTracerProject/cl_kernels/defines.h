#define PI (3.14159265359f)
#define SQ(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))

#define EPSILON (0.005f)

#define AIR_REFRACTIVE_INDEX (1.005f)
#define REFRACT_SURFACE_THICKNESS (1.0f)

#define MAX_VALUE (0xFFFFFFFF)

#define SPHERE_TYPE (0)
#define TRIANGLE_TYPE (1)

#define REFLECT_TYPE (1)
#define REFRACT_TYPE (2)

#define BVH_PLANE_COUNT (7)

// types
typedef __constant unsigned char* SKYBOX;

typedef __constant unsigned int* TRIANGLE_GRID;

typedef __constant unsigned int* TRIANGLE_GRID_COUNT;