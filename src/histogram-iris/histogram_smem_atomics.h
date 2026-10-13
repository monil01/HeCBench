template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
double run_smem_atomics(
    PixelType *d_image,
    int width,
    int height,
    unsigned int *d_hist,
    bool warmup)
{
    enum
    {
        NUM_PARTS = 1024
    };

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);

    dim3 block(32, 4);
    dim3 grid(16, 16);
    int total_blocks = grid.x * grid.y;

    // allocate partial histogram
    unsigned int *d_part_hist;
    port::allocate(&d_part_hist, total_blocks * NUM_PARTS * sizeof(unsigned int));

    dim3 block2(128);
    dim3 grid2((ACTIVE_CHANNELS * NUM_BINS + block2.x - 1) / block2.x);

    GpuTimer gpu_timer;
    gpu_timer.Start();

    const char* tag=std::is_same<PixelType,uchar1>::value ? "u1" : (std::is_same<PixelType,uchar4>::value ? "u4" : "f4");
    char name[64]; snprintf(name,sizeof(name),"hist_smem_%s",tag);
    port::launch(name,grid,block,
        d_image,
        width,
        height,
        d_part_hist);

    char accum[64]; snprintf(accum,sizeof(accum),"hist_accum_smem_%d",ACTIVE_CHANNELS);
    port::launch(accum,grid2,block2,
        d_part_hist,
        total_blocks,
        d_hist);

    gpu_timer.Stop();
    float elapsed_millis = gpu_timer.ElapsedMillis();

    port::release(d_part_hist);

    return elapsed_millis;
}
