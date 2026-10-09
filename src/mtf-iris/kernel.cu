extern "C" __global__ void mtf_kernel(const char* word, char* result, unsigned long long len)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  char list[256] = {};
  char d_list[256] = {};

  for (unsigned long long counter = 0; counter < len; counter++) {
    for (int i = 0; i < 256; i++) d_list[i] = list[i];
    const char w = word[counter];

    int iter = 0;
    while (iter < 256 && d_list[iter] != w) iter++;
    if (d_list[0] != w) {
      const int shifted = iter < 255 ? iter : 255;
      for (int i = shifted - 1; i >= 0; i--) list[i + 1] = d_list[i];
      list[0] = w;
    }
  }

  for (unsigned long long i = 0; i < len; i++) result[i] = word[i];

  for (int counter = 0; counter < 256; counter++) {
    const char symbol = list[counter];
    for (unsigned long long i = 0; i < len; i++) {
      if (result[i] == symbol) result[i] = static_cast<char>(counter);
    }
  }
}
