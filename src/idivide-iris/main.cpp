#include <iostream>
#include <cstdio>
#include <chrono>
#include "iris_support.hpp"
#include <algorithm>
#include <cstdlib>
int device;
int perf_grids = 32*1024;
int env_count(const char* key, int fallback) { const char* value = getenv(key); return value ? atoi(value) : fallback; }


#define NOW std::chrono::high_resolution_clock::now()

#include "fastdiv.h"
void throughput(const char* name) {
  int a=3,b=5,c=7,dummy=0,host[1]={0};
  iris::DMem memory(host,sizeof(host));
  void* p[] = {&a,&b,&c,&dummy,&memory};
  int info[] = {sizeof(a),sizeof(b),sizeof(c),sizeof(dummy),iris_rw};
  launch_iris(device,name,size_t(perf_grids)*256,256,5,p,info);
}
void latency(const char* name) {
  int dummy=0,host[1]={0};
  iris::DMem memory(host,sizeof(host));
  void* p[] = {&dummy,&memory};
  int info[] = {sizeof(dummy),iris_rw};
  launch_iris(device,name,size_t(perf_grids)*256,256,2,p,info);
}

// Functional test returns 1 when it fails; otherwise it returns 0
int test()
{
  const int blocks = 256;
  const int divisor_count = env_count("IDIVIDE_VALIDATION_DIVISORS",100000);
  const int divident_count = env_count("IDIVIDE_VALIDATION_DIVIDENTS",1000000);

  int grids = (divident_count + blocks - 1) / blocks;

  int buf[4] = {};
  iris::DMem memory(buf,sizeof(buf));

  std::cout << "Running functional test on " << divisor_count << " divisors, with " 
            << grids * blocks << " dividents for each divisor" << std::endl;

  for(int d = 1; d < divisor_count; ++d)
  {
    for(int sign = 1; sign >= -1; sign -= 2)
    {
      int divisor = d * sign;
      std::fill(buf,buf+4,0);
      memory.update(buf);
      void* p[] = { &divisor, &memory };
      int info[] = { sizeof(divisor), iris_rw };
      launch_iris(device,"check",size_t(grids)*blocks,blocks,2,p,info,&memory);

      if (buf[0] > 0)
      {
        std::cout << buf[0] << " wrong results, one of them is for divident " 
                  << buf[1] << ", correct quotient = " << buf[2] 
                  << ", fast computed quotient = " << buf[3] << std::endl;

        return 1;
      }
    }
  }


  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = atoi(argv[1]);

  iris::Platform platform;
  device = init_iris(platform,argc,argv);
  perf_grids = env_count("IDIVIDE_VALIDATION_GRIDS",32*1024);
  // performance evaluation after functional test is done
  if (test()) return 1;
  std::cout << "FUNCTIONAL PASS" << std::endl;

  const int grids = 32 * 1024;
  const int blocks = 256;

  // warmup may be needed for accurate performance measurement with chrono
  for (int i = 0; i < 100; i++) {
    throughput("throughput_plain");
    throughput("throughput_fast");
  }


  std::cout << "THROUGHPUT TEST" << std::endl;

  std::cout << "Benchmarking plain division by constant... ";
  auto start = NOW;

  for (int i = 0; i < repeat; i++)
    throughput("throughput_plain");


  auto end = NOW;
  std::chrono::duration<double> elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++)
    throughput("throughput_fast");


  end = NOW;
  std::chrono::duration<double> elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;

  // warmup
  for (int i = 0; i < 100; i++) {
    latency("latency_plain");
    latency("latency_fast");
  }


  std::cout << "LATENCY TEST" << std::endl;
  std::cout << "Benchmarking plain division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++)
    latency("latency_plain");


  end = NOW;
  elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++)
    latency("latency_fast");


  end = NOW;
  elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;
  const int errors = platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}
