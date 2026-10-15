#include "collectives.h"
#include "timer.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

void TestCollectivesGPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the local rank, which gets us the GPU we should be using.
    //
    // We must do this before initializing MPI, because initializing MPI requires having the right
    // GPU context, so we use environment variables from our MPI implementation to determine the
    // local rank.
    //
    // OpenMPI usually provides OMPI_COMM_WORLD_LOCAL_RANK, which we read. If you use SLURM with
    // OpenMPI, then SLURM instead provides SLURM_LOCALID. In this case, make sure to use `srun` or
    // `sbatch` and not `mpirun` to run your application.
    //
    // Remember that in order for this to work, you must have a GPU-enabled CUDA-aware MPI build.
    // Otherwise, this will result in a segfault, when MPI tries to read from a GPU memory pointer.
    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    InitCollectives(mpi_local_rank);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    cudaError_t err;

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* cpu_data = new float[size];

        float* data;
        err = cudaMalloc(&data, sizeof(float) * size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed with an error"); }

        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = std::getenv("HECBENCH_ALLREDUCE_PATTERNED") ? float(int(j%97)-48+11*mpi_rank) : 1.0f;
            }

            err = cudaMemcpy(data, cpu_data, sizeof(float) * size, cudaMemcpyHostToDevice);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            float* output;
            timer.start();
            RingAllreduce(data, size, &output);
            seconds += timer.seconds();

            err = cudaMemcpy(cpu_data, output, sizeof(float) * size, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                float expected = std::getenv("HECBENCH_ALLREDUCE_PATTERNED") ? float(mpi_size*(int(j%97)-48)+11*mpi_size*(mpi_size-1)/2) : float(mpi_size);
                if(cpu_data[j] != expected) {
                    std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << std::endl;
                    return;
                }
            }
            if(const char* prefix=std::getenv("HECBENCH_DUMP_PREFIX")) {
                char path[4096];std::snprintf(path,sizeof(path),"%s.rank%d.n%zu.iter%zu.bin",prefix,mpi_rank,size,iter);
                FILE* f=std::fopen(path,"wb");if(!f)throw std::runtime_error("dump open");
                if(std::fwrite(cpu_data,sizeof(float),size,f)!=size)throw std::runtime_error("dump write");std::fclose(f);
            }
            err = cudaFree(output);
            if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        }
        std::printf("Rank %d PASS all %zu outputs x %zu iterations, patterned=%d\n",mpi_rank,size,iters,int(std::getenv("HECBENCH_ALLREDUCE_PATTERNED")!=nullptr));
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds * 1e6 / iters
                << " us per iteration)" << std::endl;
        }

        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        delete[] cpu_data;
    }
}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {

    // Buffer sizes used for tests.
    std::vector<size_t> buffer_sizes = {
        0, 32, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864, 536870912
    };

    // Number of iterations to run for each buffer size.
    std::vector<size_t> iterations = {
        100000, 100000, 100000, 100000,
        1000, 1000, 1000, 1000,
        100, 50, 10, 1
    };

    // Opt-in bounded/parity hooks; original default lists remain unchanged.
    if(const char* p=std::getenv("HECBENCH_ALLREDUCE_TEST_LENGTH")) {buffer_sizes={std::strtoull(p,nullptr,10)};iterations={100000};}
    if(const char* p=std::getenv("HECBENCH_ALLREDUCE_MAX_ITERATIONS"))
        for(auto& count:iterations)count=std::min(count,size_t(std::strtoull(p,nullptr,10)));
    if(const char* p=std::getenv("HECBENCH_ALLREDUCE_MAX_ELEMENTS"))
        for(size_t i=buffer_sizes.size();i-- >0;)if(buffer_sizes[i]>std::strtoull(p,nullptr,10)){buffer_sizes.erase(buffer_sizes.begin()+i);iterations.erase(iterations.begin()+i);}
    TestCollectivesGPU(buffer_sizes, iterations);

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
