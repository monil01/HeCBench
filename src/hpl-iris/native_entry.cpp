// Native baseline entrypoint: original HIP GPU shim and complete HPL driver.
#include <cstdlib>
extern "C" int hpl_driver_main(int,char**);
extern "C" void hecbench_report_vendor_calls();
int main(int argc,char**argv){std::atexit(hecbench_report_vendor_calls);return hpl_driver_main(argc,argv);}
