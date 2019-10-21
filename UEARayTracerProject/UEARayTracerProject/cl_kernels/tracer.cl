#define PI (3.14159265359f)
#define SQ(x) ((x)*(x))

__kernel void TracerMain(){
    int idx = get_global_id(0);
    int idy = get_global_id(1);
}