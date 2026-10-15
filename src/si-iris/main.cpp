#include <cxxopts.hpp>
#include <fmt/core.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "io.hpp"
#include "util.hpp"
#include "host_timer.hpp"
#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"

struct Shape { int m,n,k,lda,ldb,ldc; };
static int sgemm(void* stream,unsigned,void*const* args) {
    Shape s=hecbench_vendor_arg<Shape>(args,1);
    auto a=hecbench_vendor_arg<const float*>(args,2);
    auto b=hecbench_vendor_arg<const float*>(args,3);
    auto c=hecbench_vendor_arg<float*>(args,4);
    hipblasHandle_t h;
    if(hipblasCreate(&h)!=HIPBLAS_STATUS_SUCCESS)return 1;
    if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS)return 2;
    float one=1,zero=0;
    auto status=hipblasSgemm(h,HIPBLAS_OP_N,HIPBLAS_OP_N,s.m,s.n,s.k,&one,a,s.lda,b,s.ldb,&zero,c,s.ldc);
    auto sync=hipStreamSynchronize(static_cast<hipStream_t>(stream));
    auto destroy=hipblasDestroy(h);
    return status==HIPBLAS_STATUS_SUCCESS&&sync==hipSuccess&&destroy==HIPBLAS_STATUS_SUCCESS?0:3;
}

void populateBinaryMatrix(float* input, Dataset* dataset, unsigned int start, unsigned int end);
void populateBinaryTransposeMatrix(float* input, Dataset* dataset, unsigned int partition, unsigned int start, unsigned int end);

int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "GPU Matrix multiplication set intersection"
        );

        int multiprocessorCount;
        int maxThreadsPerBlock;

        hipDeviceGetAttribute(&multiprocessorCount, hipDeviceAttributeMultiprocessorCount, 0);
        hipDeviceGetAttribute(&maxThreadsPerBlock, hipDeviceAttributeMaxThreadsPerBlock, 0);

        size_t freeDeviceMemory, totalDeviceMemory;

        hipMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);

        // arguments
        std::string input;
        std::string output;
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("partition", "Number of sets to be processed per GPU invocation", cxxopts::value<unsigned int>(partition))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("input")) {
            fmt::print("{}\n", "No input dataset given! Exiting...");
            return 1;
        }

        Dataset* d = readDataset(input);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "└{9:─^{1}}┘\n", "Dataset characteristics", 51, 25,
                "Cardinality", d->cardinality,
                "Universe", d->universe,
                "Total elements", d->totalElements, ""
        );

        d->universe++;
        partition = std::min(d->cardinality, partition);

        d->offsets = new unsigned int[d->cardinality];

        // calculate offsets
        thrust::exclusive_scan(thrust::host, d->sizes, d->sizes + d->cardinality, d->offsets, 0);

        std::vector<tile> tiles = splitToTiles(d->cardinality, partition);
        std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

        size_t combinations = partition * partition;

        unsigned long long outputMemory = runs.size() * combinations * sizeof(float);

        unsigned long long deviceMemory = (sizeof(float) * d->universe * partition * 2)
                                          + (sizeof(float) * combinations);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "└{11:─^{1}}┘\n", "Launch info", 51, 25,
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required memory (Output)", formatBytes(outputMemory),
                "Required memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        if (deviceMemory > freeDeviceMemory) {
            fmt::print("Error not enough GPU memory ({})!\nExiting...", formatBytes(freeDeviceMemory));
            return 1;
        }


        std::vector<float> counts(runs.size() * combinations);
        float* hostInput = new float[d->universe * partition];
        float* hostInvInput = new float[d->universe * partition];

        memset(hostInput, 0, d->universe * partition * sizeof(float));
        memset(hostInvInput, 0, d->universe * partition * sizeof(float));

        HostTimer hostTimer;
        Interval* device_offload = hostTimer.add("Device offload");

        setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);
        iris::Platform platform;
        platform.init(&argc,&argv,1);
        {
        iris::DMem devInput(hostInput,size_t(d->universe)*partition*sizeof(float));
        iris::DMem devInvInput(hostInvInput,size_t(d->universe)*partition*sizeof(float));
        iris::DMem devOutput(counts.data(),combinations*sizeof(float));

        unsigned int iter = 0;
        for (unsigned int i = 0; i < tiles.size(); ++i) {
            tile& A = tiles[i];

            populateBinaryMatrix(hostInput, d, A.start, A.end);

            devInput.update(hostInput);

            for (unsigned int j = i; j < tiles.size(); ++j) {
                tile& B = tiles[j];
                populateBinaryTransposeMatrix(hostInvInput, d, partition, B.start, B.end);

                devInvInput.update(hostInvInput);
                devOutput.update(counts.data()+iter*combinations);
                // Both packed transposed input and result writer have stride partition,
                // even for the shorter final tile. See README baseline repair.
                Shape shape{int(B.length),int(A.length),int(d->universe),int(partition),int(d->universe),int(partition)};
                hecbench_vendor_callback cb=sgemm;
                void* p[]={&cb,&shape,&devInvInput,&devInput,&devOutput};
                int pi[]={sizeof(cb),sizeof(shape),iris_r,iris_r,iris_rw};
                size_t gws[]={1},lws[]={1};
                iris::Task task;
                task.kernel("hecbench_vendor",1,nullptr,gws,lws,5,p,pi);
                task.flush_out(devOutput);
                if(task.submit(iris_gpu,nullptr,1))std::abort();

                // clear binary matrix to ensure correctness
                memset(hostInvInput, 0, d->universe * partition * sizeof(float));

                iter++;
            }
            // clear binary matrix to ensure correctness
            memset(hostInput, 0, d->universe * partition * sizeof(float));
        }

        }
        if(iris_error_count())std::abort();
        platform.finalize();

        HostTimer::finish(device_offload);
        hostTimer.print();

        if (!output.empty()) {
            fmt::print("Writing result to file {}\n", output);
            writeResult<float, true>(runs, partition, counts, output);
            fmt::print("Finished\n");
        }
    } catch (const cxxopts::exceptions::exception& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

void populateBinaryMatrix(float* input, Dataset* d, unsigned int start, unsigned int end)
{
    unsigned int idx = 0;
    for (unsigned int i = start; i < end; ++i) {
        for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
            input[idx * d->universe + d->elements[j]] = 1.0f;
        }
        idx++;
    }
}

void populateBinaryTransposeMatrix(float* input, Dataset* d, unsigned int partition, unsigned int start, unsigned int end)
{
    unsigned int idx = 0;
    for (unsigned int i = start; i < end; ++i) {
        for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
            input[d->elements[j] * partition + idx] = 1.0f;
        }
        idx++;
    }
}
