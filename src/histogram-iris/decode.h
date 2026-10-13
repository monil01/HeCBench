#include <hip/hip_runtime.h>
// Decode float4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__device__ __forceinline__ void DecodePixel(float4 &pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    float samples[4];
    samples[0] = pixel.x;
    samples[1] = pixel.y;
    samples[2] = pixel.z;
    samples[3] = pixel.w;

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
}

// Decode uchar4 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__device__ __forceinline__ void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    unsigned char samples[4];
    samples[0] = pixel.x;
    samples[1] = pixel.y;
    samples[2] = pixel.z;
    samples[3] = pixel.w;

    #pragma unroll
    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}

// Decode uchar1 pixel into bins
template <int NUM_BINS, int ACTIVE_CHANNELS>
__device__ __forceinline__ void DecodePixel(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    bins[0] = (unsigned int) pixel.x;
}
