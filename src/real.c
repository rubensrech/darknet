#include "real.h"

/* make_real3
 * Based on CUDA vector_functions.h
 */
__inline__ real3 make_real3(real x, real y, real z) {
    real3 t;
    t.x = x;
    t.y = y;
    t.z = z;
    return t;
}