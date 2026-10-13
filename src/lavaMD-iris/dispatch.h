#pragma once
inline void dispatch_md(dim3 grid, dim3 block, const box_str* d_box_gpu, const FOUR_VECTOR* d_rv_gpu, const fp* d_qv_gpu, FOUR_VECTOR* d_fv_gpu, const fp alpha, int dim_cpu_number_boxes) { iris_launch("md", grid, block, d_box_gpu, d_rv_gpu, d_qv_gpu, d_fv_gpu, alpha, dim_cpu_number_boxes); }
