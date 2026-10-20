#include "launch_bridge.h"
LRCallback lr_callback=nullptr;
#define main lr_source_study_main
#include "native_study.cu"
#undef main
extern "C" int lr_full_study(const char*parent,LRCallback callback){
 lr_callback=callback;char program[]="julia_authored_kernels_native_host_Thrust_hybrid";char*argv[]={program,const_cast<char*>(parent),nullptr};
 int result=lr_source_study_main(2,argv);lr_callback=nullptr;return result;
}
