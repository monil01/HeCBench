#include <iris/iris.hpp>
#include "KeccakTreeGPU.h"
void KeccakTreeGPU(tKeccakLane *h_inBuffer, iris::DMem *d_inBuffer,
 tKeccakLane *h_outBuffer, iris::DMem *d_outBuffer, iris::DMem *constants) {
 d_inBuffer->update(h_inBuffer);
 iris::Task task; void *p[]={d_inBuffer,d_outBuffer,constants};
 int info[]={iris_r,iris_w,iris_r};
 size_t g[]={NB_THREADS*NB_THREADS_BLOCKS},l[]={NB_THREADS};
 task.kernel("ker_Keccak",1,nullptr,g,l,3,p,info); task.flush_out(*d_outBuffer);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
