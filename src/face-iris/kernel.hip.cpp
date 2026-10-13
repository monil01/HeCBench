#include <hip/hip_runtime.h>
extern "C" __global__ void filter_kernel (const int*__restrict__ d_rectangles_array, 
                    int**__restrict__ d_scaled_rectangles_array, 
                    int*__restrict__ data, int width, int total_nodes)
{
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= total_nodes) return;

  int idx = gid * 12;
  for (int k = 0; k < 3; k++)
  {
    int tr_x = d_rectangles_array[idx + k * 4];
    int tr_y = d_rectangles_array[idx + 1 + k * 4];
    int tr_width = d_rectangles_array[idx + 2 + k * 4];
    int tr_height = d_rectangles_array[idx + 3 + k * 4];
    int *p0 = data + width * (tr_y) + (tr_x);
    int *p1 = data + width * (tr_y) + (tr_x + tr_width);
    int *p2 = data + width * (tr_y + tr_height) + (tr_x);
    int *p3 = data + width * (tr_y + tr_height) + (tr_x + tr_width);
    if (k < 2)
    {
      d_scaled_rectangles_array[idx + k * 4]     = p0;
      d_scaled_rectangles_array[idx + k * 4 + 1] = p1; 
      d_scaled_rectangles_array[idx + k * 4 + 2] = p2; 
      d_scaled_rectangles_array[idx + k * 4 + 3] = p3; 
    }
    else
    {
      bool z = ((tr_x == 0) && (tr_y == 0) && (tr_width == 0) && (tr_height == 0));
      d_scaled_rectangles_array[idx + k * 4]     = z ? NULL : p0;
      d_scaled_rectangles_array[idx + k * 4 + 1] = z ? NULL : p1;
      d_scaled_rectangles_array[idx + k * 4 + 2] = z ? NULL : p2;
      d_scaled_rectangles_array[idx + k * 4 + 3] = z ? NULL : p3;
    } /* end of branch if(k<2) */
  }   /* end of k loop */
}
