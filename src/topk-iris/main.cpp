#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <iris/iris.hpp>
#include "backend.h"
#include "launch.h"

int32_t main(int32_t argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <batch_size> <repeat>\n", argv[0]);
    return 1;
  }
  iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
  const int32_t batch_size = atoi(argv[1]);
  const int32_t repeat = atoi(argv[2]);

  const int32_t HIDDENSIZES[] = {3072, 4096, 8192, 16384, 32768, 65536, 131072};
  const int32_t TOPKS[] = {2048, 1024};

  for (int32_t hidden_size : HIDDENSIZES) {
    for (int32_t topk : TOPKS) {
      printf("\nbatch size: %d, hidden size: %d, topk: %d\n", batch_size, hidden_size, topk);
      int32_t total = batch_size * hidden_size;


      std::vector<float> h_x(total);
      std::vector<int32_t> h_ids(batch_size * topk);
      std::vector<float> h_out(batch_size * topk);
      iris::DMem mem_x(h_x.data(),total*sizeof(float)), mem_ids(h_ids.data(),batch_size*topk*sizeof(int32_t)), mem_values(h_out.data(),batch_size*topk*sizeof(float));
      iris::DMem *d_x=&mem_x,*d_topk_ids=&mem_ids,*d_topk_value=&mem_values;

      int32_t threads = 256;
      int32_t blocks  = (total + threads - 1) / threads;

      // initialize input
      launch_init(d_x, batch_size, hidden_size);
      launch_shuffle(d_x, batch_size, hidden_size, 5678);


      // warmup
      for (int32_t i = 0; i < 100; i++) {
        launch_topk(d_x, d_topk_ids, d_topk_value,
                                                   batch_size, hidden_size, topk);
        }

      auto start = std::chrono::steady_clock::now();

      for (int32_t i = 0; i < repeat; i++) {
        launch_topk(d_x, d_topk_ids, d_topk_value,
                                                   batch_size, hidden_size, topk);
        }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of topk : %f (us)\n", (time * 1e-3f) / repeat);


      if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
        FILE *f=fopen(path,(hidden_size==3072&&topk==2048)?"wb":"ab"); if(!f) exit(4);
        int32_t meta[]={batch_size,hidden_size,topk};
        if(fwrite(meta,sizeof(int32_t),3,f)!=3) exit(4);
        if(fwrite(h_x.data(),sizeof(float),h_x.size(),f)!=h_x.size()) exit(4);
        if(fwrite(h_ids.data(),sizeof(int32_t),h_ids.size(),f)!=h_ids.size()) exit(4);
        if(fwrite(h_out.data(),sizeof(float),h_out.size(),f)!=h_out.size()) exit(4);
        fclose(f);
      }

      bool ok = true;
      
      for (int32_t b = 0; b < batch_size; b++) {
        // partially sorted inputs from the largest
        auto start_it = h_x.begin() + b * hidden_size;
        auto end_it = h_x.begin() + (b+1) * hidden_size;
        std::partial_sort(start_it, start_it + topk, end_it, std::greater<float>());

        // sort device result
        auto o_start_it = h_out.begin() + b * topk;
        auto o_end_it = h_out.begin() + (b+1) * topk;
        std::sort(o_start_it, o_end_it, std::greater<float>());
        ok &= std::equal(start_it, start_it + topk, o_start_it);
        if (!ok) break;
      }

      printf("%s\n", ok ? "PASS" : "FAIL");

    }
  }

  platform.finalize();
  return 0;
}
