extern "C" __global__ void rle_encode_kernel(const int* in,
                                             int* unique_out,
                                             int* lengths_out,
                                             int* num_runs_out,
                                             int num_items)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (num_items <= 0) {
    *num_runs_out = 0;
    return;
  }

  int previous = in[0];
  int length = 1;
  int num_runs = 0;

  for (int i = 1; i < num_items; ++i) {
    const int current = in[i];
    if (current != previous) {
      unique_out[num_runs] = previous;
      lengths_out[num_runs] = length;
      ++num_runs;
      previous = current;
      length = 1;
    } else {
      ++length;
    }
  }

  unique_out[num_runs] = previous;
  lengths_out[num_runs] = length;
  ++num_runs;
  *num_runs_out = num_runs;
}
