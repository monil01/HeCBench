#pragma once
#include <iris/iris.hpp>
#include <cstdint>
inline void launch_init(iris::DMem *x,int32_t batch,int32_t hidden) {
 iris::Task t;void *p[]={x,&batch,&hidden};int info[]={iris_w,sizeof(batch),sizeof(hidden)};
 size_t g[]={size_t((batch*hidden+255)/256)*256},l[]={256};
 t.kernel("init_x",1,nullptr,g,l,3,p,info);t.flush_out(*x);if(t.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_shuffle(iris::DMem *x,int32_t batch,int32_t hidden,uint32_t seed) {
 iris::Task t;void *p[]={x,&batch,&hidden,&seed};int info[]={iris_rw,sizeof(batch),sizeof(hidden),sizeof(seed)};
 size_t g[]={size_t(batch)},l[]={1};
 t.kernel("shuffle_rows",1,nullptr,g,l,4,p,info);t.flush_out(*x);if(t.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
inline void launch_topk(iris::DMem *x,iris::DMem *ids,iris::DMem *values,int32_t batch,int32_t hidden,int32_t k) {
 iris::Task t;void *p[]={x,ids,values,&batch,&hidden,&k};int info[]={iris_r,iris_rw,iris_rw,sizeof(batch),sizeof(hidden),sizeof(k)};
 size_t g[]={size_t((batch*hidden+255)/256)*256},l[]={256};
 t.kernel("topk_permutation_rows",1,nullptr,g,l,6,p,info);t.flush_out(*ids);t.flush_out(*values);if(t.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)exit(3);
}
