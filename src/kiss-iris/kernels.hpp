extern "C" __global__
void kiss_kernel (std::uint64_t * const out, const std::uint64_t n, const std::uint32_t seed)
{
  const std::uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  // generate initial local seed per thread
  const std::uint32_t local_seed =
      kiss::hashers::MurmurHash<std::uint32_t>::hash(seed+tid);

  kiss::Kiss<std::uint64_t> rng {local_seed};

  // grid-stride loop
  const auto grid_stride = blockDim.x * gridDim.x;
  for(std::uint64_t i = tid; i < n; i += grid_stride)
  {
      // generate random element and write to output
      out[i] = rng.template next<std::uint64_t>();
  }
}
