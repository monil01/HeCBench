#include <iris/iris.hpp>
#include <cstring>
#include <iostream>
#include <iomanip>
#include "bench_merkle_tree.hpp"

#ifdef DEBUG
const size_t BENCH_ROUND = 1;
#else
const size_t BENCH_ROUND = 4;
#endif

int main(int argc, char** argv)
{
  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  int backend=0,type=0;char name[256]={};
  iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_type,&type,nullptr);iris_device_info(0,iris_name,name,nullptr);
  printf("Device backend=%d type=%d name=%s\n",backend,type,name);
  if(backend!=3 || type!=256 || !strstr(name,"AMD")) return 1;
  std::cout << "\nMerklize ( approach 1 ) using Rescue Prime on F(2**64 - "
               "2**32 + 1) elements\n\n";

  std::cout << std::setw(11) << "leaves"
            << "\t\t" << std::setw(15) << "total" << std::endl;

#ifdef DEBUG
  for (uint dim = 256; dim <= 256; dim <<= 1) {
#else
  for (uint dim = (1ul << 20); dim <= (1ul << 24); dim <<= 1) {
#endif
    double tm = 0;
    for (size_t i = 0; i < BENCH_ROUND; i++) {
      tm +=
        static_cast<double>(benchmark_merklize_approach_1(dim, 1ul << 5));
    }
    tm /= static_cast<double>(BENCH_ROUND);

    // time in nanoseconds
    std::cout << std::setw(11) << std::right << dim << "\t\t" << std::setw(15)
              << std::right << tm * 1e-6 << " ms" << std::endl;
  }

  platform.finalize();return 0;
}
