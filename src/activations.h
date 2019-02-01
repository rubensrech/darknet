#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "cuda.h"
#include "math.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
real activate(real x, ACTIVATION a);
real gradient(real x, ACTIVATION a);
void gradient_array(const real *x, const int n, const ACTIVATION a, real *delta);
void activate_array(real *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(real *x, int n, ACTIVATION a);
void gradient_array_gpu(real *x, int n, ACTIVATION a, real *delta);
#endif

static inline real stair_activate(real x)
{
    int n = floor(x);
    if (n%2 == 0) return CAST(floor(x/2.));
    else return CAST((x - n) + floor(x/2.));
}
static inline real hardtan_activate(real x)
{
    if (x < -1) return CAST(-1.0);
    if (x > 1) return CAST(1.0);
    return x;
}
static inline real linear_activate(real x){return x;}
static inline real logistic_activate(real x){return CAST(1./(1. + exp(-x)));}
static inline real loggy_activate(real x){return CAST(2./(1. + exp(-x)) - 1);}
static inline real relu_activate(real x){return CAST(x*(x>0));}
static inline real elu_activate(real x){return CAST((x >= 0)*x + (x < 0)*(exp(x)-1));}
static inline real selu_activate(real x){return CAST((x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1));}
static inline real relie_activate(real x){return CAST((x>0) ? x : .01*x);}
static inline real ramp_activate(real x){return CAST(x*(x>0)+.1*x);}
static inline real leaky_activate(real x){return CAST((x>0) ? x : .1*x);}
static inline real tanh_activate(real x){return CAST((exp(2*x)-1)/(exp(2*x)+1));}
static inline real plse_activate(real x)
{
    if(x < -4) return CAST(.01 * (x + 4));
    if(x > 4)  return CAST(.01 * (x - 4) + 1);
    return CAST(.125*x + .5);
}

static inline real lhtan_activate(real x)
{
    if(x < 0) return CAST(.001*x);
    if(x > 1) return CAST(.001*(x-1) + 1);
    return x;
}
static inline real lhtan_gradient(real x)
{
    if(x > 0 && x < 1) return CAST(1.0);
    return CAST(.001);
}

static inline real hardtan_gradient(real x)
{
    if (x > -1 && x < 1) return CAST(1.0);
    return CAST(0.0);
}
static inline real linear_gradient(real x){return CAST(1.0);}
static inline real logistic_gradient(real x){return CAST((1-x)*x);}
static inline real loggy_gradient(real x)
{
    real y = CAST((x+1.)/2.);
    return CAST(2*(1-y)*y);
}
static inline real stair_gradient(real x)
{
    if (floor(x) == x) return CAST(0.0);
    return CAST(1.0);
}
static inline real relu_gradient(real x){return CAST(x>0);}
static inline real elu_gradient(real x){return CAST((x >= 0) + (x < 0)*(x + 1));}
static inline real selu_gradient(real x){return CAST((x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732));}
static inline real relie_gradient(real x){return CAST((x>0) ? 1 : .01);}
static inline real ramp_gradient(real x){return CAST((x>0)+.1);}
static inline real leaky_gradient(real x){return CAST((x>0) ? 1 : .1);}
static inline real tanh_gradient(real x){return CAST(1-x*x);}
static inline real plse_gradient(real x){return CAST((x < 0 || x > 1) ? .01 : .125);}

#endif

