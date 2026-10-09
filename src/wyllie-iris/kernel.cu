#define NIL -1
#define MASK 0xFFFFFFFF

extern "C" __global__ void wyllie(long* list, const int size)
{
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < size) {
    long node;
    long next;
    while (((node = list[index]) >> 32) != NIL &&
           ((next = list[node >> 32]) >> 32) != NIL) {
      long temp = (node & MASK);
      temp += (next & MASK);
      temp += (next >> 32) << 32;
      __syncthreads();
      list[index] = temp;
    }
  }
}
