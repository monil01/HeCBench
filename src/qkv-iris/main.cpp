#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "common.h"
void matmul_forward_cpu(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            const float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                const float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}


void matmul_forward1(float* out, const float* inp, const float* weight, const float* bias, int B,int T,int C,int OC) {
    port::launch("matmul_forward_kernel1", dim3(ceil_div(B*T,16),ceil_div(OC,16)),dim3(16,16),out,inp,weight,bias,B*T,C,OC);
}
void matmul_forward2(float* out, const float* inp, const float* weight, const float* bias, int B,int T,int C,int OC) {
    port::launch("matmul_tiled",dim3(ceil_div(B*T,16),ceil_div(OC,16)),dim3(16,16),out,inp,weight,bias,B*T,C,OC,0);
    if (bias) port::launch("add_bias",dim3(ceil_div(B*T*OC,256)),dim3(256),out,bias,B,T,OC);
}
void matmul_forward3(float* out, const float* inp, const float* weight, const float* bias, int B,int T,int C,int OC) {
    port::launch("matmul_tiled",dim3(ceil_div(B*T,16),ceil_div(OC,16)),dim3(16,16),out,inp,weight,bias,B*T,C,OC,1);
}
void matmul_forward4(float* out, const float* inp, const float* weight, const float* bias, int B,int T,int C,int OC) {
    if ((B*T)%128 || OC%128 || C%32) { fprintf(stderr,"Kernel 4 requires BT/OC multiples of 128 and C multiple of 32\n"); exit(1); }
    port::launch("matmul_forward_kernel4",dim3(ceil_div(B*T,128),ceil_div(OC,128)),dim3(16,16),out,inp,weight,bias,C,OC);
}
// kernel version dispatch
void matmul_forward(int kernel_num,
                    float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    switch (kernel_num) {
        case 1:
            matmul_forward1(out, inp, weight, bias, B, T, C, OC);
            break;
        case 2:
            matmul_forward2(out, inp, weight, bias, B, T, C, OC);
            break;
        case 3:
            matmul_forward3(out, inp, weight, bias, B, T, C, OC);
            break;
        case 4:
            matmul_forward4(out, inp, weight, bias, B, T, C, OC);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

int main(int argc, char **argv) {

    int B = 4;
    int T = 1024;
    int C = 768;
    int OC = 768 * 3;

    if (const char* p = getenv("HECBENCH_QKV_B")) B = atoi(p);
    if (const char* p = getenv("HECBENCH_QKV_T")) T = atoi(p);
    if (const char* p = getenv("HECBENCH_QKV_C")) C = atoi(p);
    if (const char* p = getenv("HECBENCH_QKV_OC")) OC = atoi(p);
    if (B <= 0 || T <= 0 || C <= 0 || OC <= 0) { fprintf(stderr, "Invalid QKV dimensions\n"); return 1; }

    port::Session session(&argc, &argv);
    printf("enable_tf32: 0\n");

    // IRIS PolicyRandom initialization reseeds libc rand; restore original seed.
    srand(0);
    // create host memory of random numbers
    float* out = (float*)malloc(B * T * OC * sizeof(float));
    float* inp = make_random_float(B * T * C);
    float* weight = make_random_float(OC * C);
    float* bias = make_random_float(OC);

    // move to GPU
    float* d_out;
    float* d_inp;
    float* d_weight;
    float* d_bias;
    hipCheck(port::allocate(&d_out, B * T * OC * sizeof(float)));
    hipCheck(port::allocate(&d_inp, B * T * C * sizeof(float)));
    hipCheck(port::allocate(&d_weight, C * OC * sizeof(float)));
    hipCheck(port::allocate(&d_bias, OC * sizeof(float)));
    hipCheck(port::copy(d_inp, inp, B * T * C * sizeof(float), port::h2d));
    hipCheck(port::copy(d_weight, weight, C * OC * sizeof(float), port::h2d));
    hipCheck(port::copy(d_bias, bias, OC * sizeof(float), port::h2d));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    matmul_forward_cpu(out, inp, weight, bias, B, T, C, OC);

    matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC);
    validate_result(d_out, out, "out", B * T * OC, 1e-1f);

    printf("All results match. Starting benchmarks.\n\n");

    int repeat_times = 100;
    if (const char* p = getenv("HECBENCH_QKV_REPEAT")) repeat_times = atoi(p);
    if (repeat_times <= 0) return 1;
    float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
                                          kernel_num, d_out, d_inp, d_weight, d_bias,
                                          B, T, C, OC);

    float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
    printf("time %.4f ms | tflops %.2f\n", elapsed_time, tflops);

    // free memory
    free(out);
    free(inp);
    free(weight);
    free(bias);
    hipCheck(port::release(d_out));
    hipCheck(port::release(d_inp));
    hipCheck(port::release(d_weight));
    hipCheck(port::release(d_bias));

    return 0;
}
