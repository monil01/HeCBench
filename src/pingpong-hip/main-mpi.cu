#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>
#include <mpi.h>

// Macro for checking errors in HIP API calls
#define hipErrorCheck(call)                                                              \
  do{                                                                                    \
    hipError_t cuErr = call;                                                             \
    if(hipSuccess != cuErr){                                                             \
      printf("HIP Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(cuErr)); \
      exit(0);                                                                           \
    }                                                                                    \
  }while(0)


__global__
void test(double *d, const long int n) {
  for (long i = blockDim.x * blockIdx.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    d[i] = d[i] + 1;
  }
}

static void observe_gpu(double* device,long n,int rank,const char* stage) {
  const char* prefix=getenv("HECBENCH_DUMP_PREFIX");if(!prefix)return;
  double* host=(double*)malloc(n*sizeof(double));if(!host)exit(1);
  hipErrorCheck(hipMemcpy(host,device,n*sizeof(double),hipMemcpyDeviceToHost));
  const bool pattern=getenv("HECBENCH_PINGPONG_PATTERNED")!=nullptr;
  for(long j=0;j<n;++j)if(!std::isfinite(host[j])||host[j]!=double((pattern?int(j%97)-48:0)+5)){printf("ERROR: whole-output mpi verification failed at %ld\n",j);exit(1);}
  char path[4096];snprintf(path,sizeof(path),"%s.rank%d.n%ld.%s.bin",prefix,rank,n,stage);FILE* f=fopen(path,"wb");if(!f)exit(1);
  if(fwrite(host,sizeof(double),n,f)!=size_t(n))exit(1);fclose(f);free(host);
  printf("MPI Rank %d PASS all %ld %s outputs\n",rank,n,stage);
}

int main(int argc, char *argv[])
{
  /* -------------------------------------------------------------------------------------------
     MPI Initialization
     --------------------------------------------------------------------------------------------*/
  // GPU-aware MPI must see the intended local device during initialization.
  const char* local=getenv("OMPI_COMM_WORLD_LOCAL_RANK");if(!local)local=getenv("SLURM_LOCALID");
  if(local)hipErrorCheck(hipSetDevice(atoi(local)));
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Status stat;

  if(size != 2){
    if(rank == 0){
      printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
    }
    MPI_Finalize();
    exit(0);
  }

  // Map MPI ranks to GPUs
  int num_devices = 0;
  hipErrorCheck( hipGetDeviceCount(&num_devices) );
  hipErrorCheck( hipSetDevice(rank % num_devices) );

  //   Loop from 65536 B to 1 GB
  int maximum=27;if(const char* cap=getenv("HECBENCH_PINGPONG_MAX_EXP"))maximum=std::min(maximum,atoi(cap));
  const char* custom=getenv("HECBENCH_PINGPONG_TEST_LENGTH");
  for(int i=16; i<=maximum; i++){
    if(custom&&i>16)break;

    long int N = custom ? strtol(custom,nullptr,10) : (1L << i);

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double));
    hipErrorCheck( hipMalloc((void**)&d_A, N*sizeof(double)) );
    hipErrorCheck( hipMemset(d_A, 0, N*sizeof(double)) );
    if(getenv("HECBENCH_PINGPONG_PATTERNED")) {
      for(long j=0;j<N;++j)h_A[j]=double(int(j%97)-48+7*rank);
      hipErrorCheck(hipMemcpy(d_A,h_A,N*sizeof(double),hipMemcpyHostToDevice));
    }
    hipErrorCheck( hipDeviceSynchronize() );

    const int tag1 = 10;
    const int tag2 = 20;

    int loop_count = 50;
    if(const char* cap=getenv("HECBENCH_PINGPONG_MAX_ITERATIONS"))loop_count=std::min(loop_count,atoi(cap));

    // Warm-up and validate MPI pingpong
    for(int i=1; i<=5; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        test<<<1024, 256>>>(d_A, N);
        hipErrorCheck( hipDeviceSynchronize() );
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }
    if(rank == 0) {
      hipErrorCheck(hipMemcpy(h_A, d_A, N*sizeof(double), hipMemcpyDeviceToHost));
      for (long int i = 0; i < N; i++) {
        if(h_A[i] != double((getenv("HECBENCH_PINGPONG_PATTERNED")?int(i%97)-48:0)+5)) {
          printf("ERROR: MPI pingpong test failed\n");
          break;
        }
      }
    }

    observe_gpu(d_A,N,rank,"warm");
    free(h_A);

    // Time loop_count iterations of data transfer size 8*N bytes
    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    for(int i=1; i<=loop_count; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    long int num_B = 8*N;
    double num_GB = (double)num_B / 1.0e9;
    double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

    if(rank == 0)
      printf("MPI : Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    observe_gpu(d_A,N,rank,"timed");
    hipErrorCheck( hipFree(d_A) );
  }

  MPI_Finalize();

  return 0;
}
