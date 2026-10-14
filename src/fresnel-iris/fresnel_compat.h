#ifndef FRESNEL_COMPAT_H
#define FRESNEL_COMPAT_H

#ifndef __CUDACC__
#ifndef __HIPCC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif
#endif

#endif
