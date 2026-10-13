#include <chrono>
#include <cstdio>
#include "iris_support.hpp"
#include <cstdint>
#include <cstdlib>
#include <vector>

// generation of gigabytes of uniform random values

int main(int argc, char* argv[])
{
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      return 1;
    }
    const int repeat = atoi(argv[1]);

    // define the data types to be generated
    using data_t = std::uint64_t;


    const std::uint64_t n = getenv("KISS_VALIDATION_SIZE") ?
        strtoull(getenv("KISS_VALIDATION_SIZE"),nullptr,10) : 1UL << 28;

    static constexpr std::uint32_t seed = 42;

    iris::Platform platform;
    int device=init_iris(platform,argc,argv);
    std::vector<data_t> storage(n,0);
    data_t* data_h=storage.data();
    iris::DMem data_d(data_h,sizeof(data_t)*n);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      void* p[] = { &data_d, const_cast<std::uint64_t*>(&n), const_cast<std::uint32_t*>(&seed) };
      int info[] = { iris_w, sizeof(n), sizeof(seed) };
      launch_iris(device,"kiss_kernel",4096*256,256,3,p,info);
    }


    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

    flush_iris(device,data_d);

    printf("\nThe first 10 random numbers:\n");
    for(std::uint64_t i = 0; i < 10; i++)
    {
      printf("%lu\n", data_h[i]);
    }

    if (const char* path=getenv("KISS_VALIDATION_DUMP")) {
      FILE* fp=fopen(path,"wb");
      if(!fp) return 1;
      const size_t count=fwrite(data_h,sizeof(data_t),n,fp);
      fclose(fp);
      if(count!=n) return 1;
    }
    const int errors=platform.error_count();
    check_iris(platform.finalize());
    return errors ? 1 : 0;
}
