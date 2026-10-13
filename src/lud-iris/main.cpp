#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include "iris_support.h"
#include "common.h"

#define BLOCK_SIZE 16


static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

int main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;

  while ((opt = getopt_long(argc, argv, "::vs:i:",
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
        input_file = optarg;
        break;
      case 'v':
        do_verify = 1;
        break;
      case 's':
        matrix_dim = atoi(optarg);
        if (matrix_dim <= 0) {
          printf("Matrix dimension must be positive!\n");
          exit(EXIT_FAILURE);
        }
        if (matrix_dim % 16 != 0) {
          printf("Matrix dimension of %d not supported by the benchmark\n", matrix_dim);
          exit(EXIT_FAILURE);
        }
        printf("Generate input matrix internally, size=%d\n", matrix_dim);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }

  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }

  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  /* beginning of timing point */
  auto start = std::chrono::steady_clock::now();

  if(matrix_dim<=0 || matrix_dim%16) return 1;
  iris::Platform platform;
  iris_check(platform.init(&argc,&argv,true),"init");
  int device=iris_hip_device();
  size_t matrix_size_bytes=static_cast<size_t>(matrix_dim)*matrix_dim*sizeof(float);
  const size_t dim=matrix_dim;
  iris::DMem d_m(m,matrix_size_bytes);
  auto launch = [&](const char* name,int offset,int ndims,size_t* gws,size_t* lws) {
    iris::Task task;
    void* args[]={&d_m,(void*)&dim,&offset};
    int info[]={iris_rw,sizeof(dim),sizeof(offset)};
    iris_check(task.kernel(name,ndims,nullptr,gws,lws,3,args,info),"kernel");
    iris_check(task.submit(device,nullptr,true),"HIP task");
  };
  size_t diag_local=BLOCK_SIZE, diag_global=BLOCK_SIZE;

  int i=0;
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  auto kstart = std::chrono::steady_clock::now();

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
    launch("lud_diagonal",i,1,&diag_global,&diag_local);
    size_t groups=(matrix_dim-i)/BLOCK_SIZE-1;
    size_t perimeter_local=2*BLOCK_SIZE, perimeter_global=groups*perimeter_local;
    launch("lud_perimeter",i,1,&perimeter_global,&perimeter_local);
    size_t internal_local[]={BLOCK_SIZE,BLOCK_SIZE};
    size_t internal_global[]={groups*BLOCK_SIZE,groups*BLOCK_SIZE};
    launch("lud_internal",i,2,internal_global,internal_local);
  }
  launch("lud_diagonal",i,1,&diag_global,&diag_local);

  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Total kernel execution time : %lf (s)\n", ktime * 1e-9);

  iris::Task flush;
  iris_check(flush.flush_out(d_m),"flush matrix");
  iris_check(flush.submit(device,nullptr,true),"HIP flush");


  /* end of timing point */
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time (s): %lf\n", time * 1e-9);

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim);
    printf("%s\n", lud_verification_errors ? "FAIL" : "PASS");
    free(mm);
  }

  if(std::getenv("HECBENCH_DUMP")) {
    FILE* fp=fopen("lud_result.bin","wb");
    if(!fp || fwrite(m,sizeof(float),static_cast<size_t>(matrix_dim)*matrix_dim,fp)!=static_cast<size_t>(matrix_dim)*matrix_dim) return 1;
    fclose(fp);
  }
  free(m);
  iris_check(platform.finalize(),"finalize");
  return do_verify && lud_verification_errors ? 1 : 0;
}
