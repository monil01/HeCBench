#include <iris/iris.hpp>
#include "dmem_adapter.hpp"
extern int iris_logic_main(int,char**);
int main(int argc,char**argv){
  setenv("IRIS_ARCHS","HIP",1);setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",1);
  iris::Platform platform;if(platform.init(&argc,&argv,1))return 1;
  int result=iris_logic_main(argc,argv);
  iris_dmem::cleanup();result=result || iris_error_count();platform.finalize();return result;
}
