/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <chrono>
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include "backend.h"
#include "launch.h"

#include "kernel.h"
#include "support/partitioner.h"
#include "support/verify.h"

// Params ---------------------------------------------------------------------
struct Params {

  int         device;
  int         n_gpu_threads;
  int         n_threads;
  int         n_warmup;
  int         n_reps;
  float       alpha;
  const char *file_name;
  const char *comparison_file;
  int         display = 0;

  Params(int argc, char **argv) {
    device          = 0;
    n_gpu_threads   = 16;
    n_threads       = 4;
    n_warmup        = 10;
    n_reps          = 100;
    alpha           = 0.2;
    file_name       = "input/peppa/";
    comparison_file = "output/peppa/";
    int opt;
    while((opt = getopt(argc, argv, "hd:i:t:w:r:a:f:c:x")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads   = atoi(optarg); break;
        case 't': n_threads       = atoi(optarg); break;
        case 'w': n_warmup        = atoi(optarg); break;
        case 'r': n_reps          = atoi(optarg); break;
        case 'a': alpha           = atof(optarg); break;
        case 'f': file_name       = optarg; break;
        case 'c': comparison_file = optarg; break;
        case 'x': display         = 1; break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
    if(alpha == 0.0) {
      assert(n_gpu_threads > 0 && "Invalid # of device threads!");
    } else if(alpha == 1.0) {
      assert(n_threads > 0 && "Invalid # of host threads!");
    } else if(alpha > 0.0 && alpha < 1.0) {
      assert(n_gpu_threads > 0 && "Invalid # of device threads!");
      assert(n_threads > 0 && "Invalid # of host threads!");
    } else {
      assert((n_gpu_threads > 0 || n_threads > 0) && "Invalid # of host + device workers!");
    }
#ifndef CHAI_OPENCV
    assert(display != 1 && "Compile with CHAI_OPENCV");
#endif
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./cedd [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=16)"
        "\n    -t <T>    # of host threads (default=4)"
        "\n    -w <W>    # of untimed warmup iterations (default=10)"
        "\n    -r <R>    # of timed repetition iterations (default=100)"
        "\n"
        "\nData-partitioning-specific options:"
        "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
        "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -f <F>    folder containing input video files (default=input/peppa/)"
        "\n    -c <C>    folder containing comparison files (default=output/peppa/)"
        "\n    -x        display output video (with CHAI_OPENCV)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char** all_gray_frames, 
		int &rows, int &cols, int &in_size, const Params &p) {

  for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

    char FileName[100];
    sprintf(FileName, "%s%d.txt", p.file_name, task_id);

    FILE *fp = fopen(FileName, "r");
    if(fp == NULL) {
      fprintf (stderr, "Failed to open the file %s. Exit\n.", FileName);
      exit(EXIT_FAILURE);
    }

    fscanf(fp, "%d\n", &rows);
    fscanf(fp, "%d\n", &cols);

    in_size = rows * cols * sizeof(unsigned char);
    all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        unsigned int pixel; if(fscanf(fp, "%u ", &pixel)!=1 || pixel>255) exit(4);
        all_gray_frames[task_id][i * cols + j] = static_cast<unsigned char>(pixel);
      }
    }
    fclose(fp);
  }
}

int main(int argc, char **argv) {

  iris::Platform platform; platform.init(&argc,&argv,1); prove_backend();
  Params      p(argc, argv);

  // The maximum number of GPU threads is 1024 for certain GPUs
  const int max_gpu_threads = 256;
  assert(p.n_gpu_threads * p.n_gpu_threads <= max_gpu_threads && \
   "The thread block size is greater than the maximum thread block size");

  // read data from an 'input' directory which must be available
  const int n_frames = p.n_warmup + p.n_reps;
  unsigned char **all_gray_frames = 
    (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
  int     rows, cols, in_size;
  read_input(all_gray_frames, rows, cols, in_size, p);

  unsigned char **all_out_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
  for(int i = 0; i < n_frames; i++) {
    all_out_frames[i] = (unsigned char *)malloc(in_size);
  }
  std::atomic_int *worklist = (std::atomic_int *)malloc(sizeof(std::atomic_int));
  if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
    worklist[0].store(0);
  }

  unsigned char* cpu_in_out = (unsigned char *)malloc(in_size);

  unsigned char *h_interm_cpu_proxy = (unsigned char *)malloc(in_size);
  unsigned char *h_theta_cpu_proxy  = (unsigned char *)malloc(in_size);

  std::vector<unsigned char> gpu_in_out(in_size),gpu_interm(in_size),gpu_theta(in_size);
  iris::DMem d_in_out(gpu_in_out.data(),in_size);
  iris::DMem d_interm_gpu_proxy(gpu_interm.data(),in_size);
  iris::DMem d_theta_gpu_proxy(gpu_theta.data(),in_size);

  CoarseGrainPartitioner partitioner = partitioner_create(n_frames, p.alpha, worklist);
  std::vector<std::thread> proxy_threads;

  auto t1 = std::chrono::high_resolution_clock::now();

  proxy_threads.push_back(std::thread([&]() {

      for(int task_id = gpu_first(&partitioner); gpu_more(&partitioner); task_id = gpu_next(&partitioner)) {

        // Copy next frame to device
        memcpy(gpu_in_out.data(),all_gray_frames[task_id],in_size);
        d_in_out.update(gpu_in_out.data());
     
        int threads = p.n_gpu_threads;
        Dim3 grid ((cols-2)/threads, (rows-2)/threads);
        Dim3 block (threads, threads);
        int smem_size = (threads+2)*(threads+2)*sizeof(int); 

        // call GAUSSIAN KERNEL
        launch_gaussian(grid,block,d_in_out,d_interm_gpu_proxy,rows,cols);

        // call SOBEL KERNEL
        launch_sobel(grid,block,d_interm_gpu_proxy,d_in_out,d_theta_gpu_proxy,rows,cols);

        // call NON-MAXIMUM SUPPRESSION KERNEL
        launch_nonmax(grid,block,d_in_out,d_interm_gpu_proxy,d_theta_gpu_proxy,rows,cols);

        // call HYSTERESIS KERNEL
        launch_hyst(grid,block,d_interm_gpu_proxy,d_in_out,rows,cols);

        // Copy from Device
        memcpy(all_out_frames[task_id],gpu_in_out.data(),in_size);
      }

      for(int task_id = cpu_first(&partitioner); cpu_more(&partitioner); task_id = cpu_next(&partitioner)) {

        // Next frame
        memcpy(cpu_in_out, all_gray_frames[task_id], in_size);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, cpu_in_out, h_interm_cpu_proxy, h_theta_cpu_proxy,
            rows, cols, p.n_threads, task_id);
        main_thread.join();

        memcpy(all_out_frames[task_id], cpu_in_out, in_size);

      }
  }));
  std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });

  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Total time %lf secs \n", total_time / 1.0e6);


#ifdef CHAI_OPENCV
  // Display the result
  if(p.display){
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
      cv::Mat out_frame = cv::Mat(rows, cols, CV_8UC1);
      memcpy(out_frame.data, all_out_frames[rep], in_size);
      if(!out_frame.empty())
        imshow("canny", out_frame);
      if(cv::waitKey(30) >= 0)
        break;
    }
  }
#endif

  // Verify answer
  int status = verify(all_out_frames, in_size, p.comparison_file, 
		  p.n_warmup + p.n_reps, rows, cols, rows, cols);

  if(const char *path=getenv("HECBENCH_VALIDATION_DUMP")) {
    FILE *file=fopen(path,"wb"); if(!file)exit(5);
    int meta[]={n_frames,rows,cols};fwrite(meta,sizeof(int),3,file);
    for(int i=0;i<n_frames;++i)fwrite(all_out_frames[i],1,in_size,file);
    fclose(file);
  }
  // Release buffers

  free(cpu_in_out);
  free(h_interm_cpu_proxy);
  free(h_theta_cpu_proxy);
  for(int i = 0; i < n_frames; i++) {
    free(all_gray_frames[i]);
  }
  free(all_gray_frames);
  for(int i = 0; i < n_frames; i++) {
    free(all_out_frames[i]);
  }
  free(all_out_frames);
  free(worklist);

  if (status == 0) printf("PASS\n");
  platform.finalize();
  return status==0?0:1;
}
