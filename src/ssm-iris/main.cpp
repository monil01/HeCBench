#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include "reference.h"
#ifndef SSM_BATCH
#define SSM_BATCH 32
#endif
#ifndef SSM_DIM
#define SSM_DIM 2048
#endif
#ifndef SSM_SEQLEN
#define SSM_SEQLEN 1024
#endif
static constexpr int kBatch=SSM_BATCH,kDim=SSM_DIM,kDstate=8,kSeqLen=SSM_SEQLEN,kNT=128,kNI=4;
#define HOST_CALL(call) do { call; } while(0)
struct Buffer {
  std::vector<float> host;
  iris::DMem mem;
  explicit Buffer(size_t bytes):host(bytes/sizeof(float)),mem(host.data(),bytes){}
};
static void buffer_create(Buffer** p,size_t bytes){*p=new Buffer(bytes);}
static void buffer_free(Buffer* p){delete p;}
static void buffer_input(Buffer* p,const float* h,size_t bytes){std::memcpy(p->host.data(),h,bytes);p->mem.update(p->host.data());}
static void buffer_output(float* h,Buffer* p,size_t bytes){std::memcpy(h,p->host.data(),bytes);}
static void buffer_zero(Buffer* p,int value,size_t bytes){std::memset(p->host.data(),value,bytes);p->mem.update(p->host.data());}
static void dump(const char* prefix,const char* name,const float* h,size_t n){
  std::string path=std::string(prefix)+"_"+name+".bin";
  FILE* f=std::fopen(path.c_str(),"wb");if(!f || std::fwrite(h,sizeof(float),n,f)!=n)std::exit(2);std::fclose(f);
}
struct SSMParamsBase {
  int batch,dim,seqlen,dstate,n_chunks;
  Buffer *u_ptr,*delta_ptr,*A_ptr,*B_ptr,*C_ptr,*D_ptr,*delta_bias_ptr,*z_ptr,*ssm_states_ptr,*out_ptr;
  int u_batch_stride,u_d_stride,B_batch_stride,B_dstate_stride,states_batch_stride,states_d_stride;
};
static void launch_base(Buffer* u,Buffer* delta,Buffer* a,Buffer* b,Buffer* c,Buffer* d,Buffer* dbias,
                        Buffer* z,bool softplus,int batch,int dim,int dstate,int seqlen,Buffer* states,Buffer* out){
  int tx=(dim<1024)?dim:1024;
  Buffer* gate=z?z:u; // Unused binding when the original optional gate is null.
  size_t global[]={static_cast<size_t>(batch*tx),static_cast<size_t>((dim+tx-1)/tx)},local[]={static_cast<size_t>(tx),1};
  void* args[]={&u->mem,&delta->mem,&a->mem,&b->mem,&c->mem,&d->mem,&dbias->mem,&gate->mem,
                &softplus,&batch,&dim,&dstate,&seqlen,&states->mem,&out->mem};
  int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,
              sizeof(softplus),sizeof(batch),sizeof(dim),sizeof(dstate),sizeof(seqlen),iris_rw,iris_w};
  iris::Task task;task.kernel("selective_scan_base",2,nullptr,global,local,15,args,info);
  task.flush_out(states->mem);task.flush_out(out->mem);task.submit(iris_gpu,nullptr,true);
}
static void launch_kernel(SSMParamsBase& p,bool has_z,bool softplus,bool even_len,Buffer* ca,Buffer* cb){
  std::string name="ssm_";name+=has_z?'1':'0';name+=softplus?'1':'0';name+=even_len?'1':'0';
  Buffer* gate=p.z_ptr?p.z_ptr:p.u_ptr;
  size_t global[]={static_cast<size_t>(p.batch*p.dim*kNT),static_cast<size_t>(p.n_chunks)},local[]={kNT,1};
  void* args[]={&p.u_ptr->mem,&p.delta_ptr->mem,&p.A_ptr->mem,&p.B_ptr->mem,&p.C_ptr->mem,
                &p.D_ptr->mem,&p.delta_bias_ptr->mem,&gate->mem,&p.ssm_states_ptr->mem,&p.out_ptr->mem,
                &ca->mem,&cb->mem,&p.batch,&p.dim,&p.seqlen,&p.dstate,&p.n_chunks};
  int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw,iris_w,iris_rw,iris_rw,
              sizeof(int),sizeof(int),sizeof(int),sizeof(int),sizeof(int)};
  iris::Task task;task.kernel(name.c_str(),2,nullptr,global,local,17,args,info);
  task.flush_out(p.ssm_states_ptr->mem);task.flush_out(p.out_ptr->mem);task.submit(iris_gpu,nullptr,true);
}
static void fill_rand(float* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; ++i)
        buf[i] = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}
static float max_abs_err(const float* a, const float* b, int n) {
    float e = 0.f;
    for (int i = 0; i < n; ++i) e = fmaxf(e, fabsf(a[i] - b[i]));
    return e;
}
static void print_result(const char* label, float err_y, float err_s, float tol) {
    printf("  %-22s  err_y=%.3e  err_s=%.3e  %s\n",
           label, err_y, err_s,
           (err_y < tol && err_s < tol) ? "PASS" : "FAIL");
}

int main(int argc, char **argv)
{
    if (argc != 2) {
      printf("Usage: %s <repeat>\n", argv[0]);
      return 1;
    }
    iris::Platform platform;
    platform.init(&argc, &argv, 1);
    const int repeat = atoi(argv[1]);
    const int batch  = kBatch;
    const int dim    = kDim;
    const int dstate = kDstate;
    const int seqlen = kSeqLen;
    const bool use_z = true; // test with SiLU gate enabled

    printf("  batch=%d  dim=%d  dstate=%d  seqlen=%d  z=%s\n\n",
           batch, dim, dstate, seqlen, use_z ? "yes" : "no");

    int N_udz  = batch * dim * seqlen;
    int N_A    = dim * dstate;
    int N_BC   = batch * dstate * seqlen;
    int N_D    = dim;
    int N_st   = batch * dim * dstate;
    int N_y    = N_udz;

    size_t SZ  = sizeof(float);

    float *h_u      = (float*)malloc(N_udz * SZ);
    float *h_delta  = (float*)malloc(N_udz * SZ);
    float *h_A      = (float*)malloc(N_A   * SZ);
    float *h_B      = (float*)malloc(N_BC  * SZ);
    float *h_C      = (float*)malloc(N_BC  * SZ);
    float *h_D      = (float*)malloc(N_D   * SZ);
    float *h_dbias  = (float*)malloc(N_D   * SZ);
    float *h_z      = (float*)malloc(N_udz * SZ);
    float *h_states0= (float*)malloc(N_st  * SZ);   // initial state (zeros)
    float *h_s_ref  = (float*)malloc(N_st  * SZ);
    float *h_s_n    = (float*)malloc(N_st  * SZ);   // base kernel states
    float *h_s_v    = (float*)malloc(N_st  * SZ);   // vllm-style  kernel states
    float *h_y_ref  = (float*)malloc(N_y   * SZ);
    float *h_y_n    = (float*)malloc(N_y   * SZ);
    float *h_y_v    = (float*)malloc(N_y   * SZ);

    srand(19937);
    fill_rand(h_u,     N_udz, -1.f,  1.f);
    fill_rand(h_delta, N_udz, -1.f,  1.f);
    fill_rand(h_A,     N_A,   -1.f,  0.f);   // A <= 0 for stability
    fill_rand(h_B,     N_BC,  -1.f,  1.f);
    fill_rand(h_C,     N_BC,  -1.f,  1.f);
    fill_rand(h_D,     N_D,   -1.f,  1.f);
    fill_rand(h_dbias, N_D,   -1.f,  1.f);
    fill_rand(h_z,     N_udz, -2.f,  2.f);
    memset(h_states0, 0, N_st * SZ);

    // reference
    memcpy(h_s_ref, h_states0, N_st * SZ);
    selective_scan_ref(h_u, h_delta, h_A, h_B, h_C, h_D, h_dbias,
                       use_z ? h_z : nullptr, /*delta_softplus=*/true,
                       batch, dim, dstate, seqlen,
                       h_s_ref, h_y_ref);

    Buffer *d_u, *d_delta, *d_A, *d_B, *d_C, *d_D, *d_dbias, *d_z;
    Buffer *d_states, *d_y;

    HOST_CALL(buffer_create(&d_u,      N_udz * SZ));
    HOST_CALL(buffer_create(&d_delta,  N_udz * SZ));
    HOST_CALL(buffer_create(&d_A,      N_A   * SZ));
    HOST_CALL(buffer_create(&d_B,      N_BC  * SZ));
    HOST_CALL(buffer_create(&d_C,      N_BC  * SZ));
    HOST_CALL(buffer_create(&d_D,      N_D   * SZ));
    HOST_CALL(buffer_create(&d_dbias,  N_D   * SZ));
    HOST_CALL(buffer_create(&d_z,      N_udz * SZ));
    HOST_CALL(buffer_create(&d_states, N_st  * SZ));
    HOST_CALL(buffer_create(&d_y,      N_y   * SZ));

    HOST_CALL(buffer_input(d_u,     h_u,     N_udz * SZ));
    HOST_CALL(buffer_input(d_delta, h_delta, N_udz * SZ));
    HOST_CALL(buffer_input(d_A,     h_A,     N_A   * SZ));
    HOST_CALL(buffer_input(d_B,     h_B,     N_BC  * SZ));
    HOST_CALL(buffer_input(d_C,     h_C,     N_BC  * SZ));
    HOST_CALL(buffer_input(d_D,     h_D,     N_D   * SZ));
    HOST_CALL(buffer_input(d_dbias, h_dbias, N_D   * SZ));
    HOST_CALL(buffer_input(d_z,     h_z,     N_udz * SZ));
    HOST_CALL(buffer_input(d_states, h_states0, N_st * SZ));

    int tx = (dim < 1024) ? dim : 1024;
    // Original base grid is reconstructed by launch_base.
    launch_base(
        d_u, d_delta, d_A, d_B, d_C, d_D, d_dbias,
        use_z ? d_z : nullptr, /*delta_softplus=*/true,
        batch, dim, dstate, seqlen, d_states, d_y
    );

    HOST_CALL(buffer_output(h_y_n, d_y,      N_y  * SZ));
    HOST_CALL(buffer_output(h_s_n, d_states, N_st * SZ));

    // reset states
    HOST_CALL(buffer_input(d_states, h_states0, N_st * SZ));
    const int CHUNK  = kNT * kNI;
    int n_chunks = (seqlen + CHUNK - 1) / CHUNK;
    bool even_len = (seqlen % CHUNK == 0);

    int N_carry = batch * dim * dstate * n_chunks;
    Buffer *d_carry_a, *d_carry_b;
    HOST_CALL(buffer_create(&d_carry_a, N_carry * SZ));
    HOST_CALL(buffer_zero(d_carry_a, 0, N_carry * SZ));
    HOST_CALL(buffer_create(&d_carry_b, N_carry * SZ));
    HOST_CALL(buffer_zero(d_carry_b, 0, N_carry * SZ));

    SSMParamsBase p;
    p.batch   = batch;  p.dim   = dim;
    p.seqlen  = seqlen; p.dstate = dstate;
    p.n_chunks = n_chunks;
    p.u_ptr   = d_u;     p.delta_ptr      = d_delta;
    p.A_ptr   = d_A;     p.B_ptr          = d_B;
    p.C_ptr   = d_C;     p.D_ptr          = d_D;
    p.delta_bias_ptr = d_dbias;
    p.z_ptr   = use_z ? d_z : nullptr;
    p.ssm_states_ptr = d_states;
    p.out_ptr = d_y;
    // Strides (elements)
    p.u_batch_stride     = dim * seqlen;
    p.u_d_stride         = seqlen;
    p.B_batch_stride     = dstate * seqlen;
    p.B_dstate_stride    = seqlen;
    p.states_batch_stride= dim * dstate;
    p.states_d_stride    = dstate;

    launch_kernel(p, use_z, /*softplus=*/true, even_len, d_carry_a, d_carry_b);

    HOST_CALL(buffer_output(h_y_v, d_y,      N_y  * SZ));
    HOST_CALL(buffer_output(h_s_v, d_states, N_st * SZ));
    HOST_CALL(buffer_free(d_carry_a));
    HOST_CALL(buffer_free(d_carry_b));

    if (const char* prefix=std::getenv("SSM_VERIFY_PREFIX")) {
      dump(prefix,"base_y",h_y_n,N_y); dump(prefix,"base_s",h_s_n,N_st);
      dump(prefix,"scan_y",h_y_v,N_y); dump(prefix,"scan_s",h_s_v,N_st);
    }
    const float TOL = 1e-3f;
    printf("  Results vs reference (tolerance=%.0e):\n", TOL);
    print_result("Base kernel",
                 max_abs_err(h_y_ref, h_y_n, N_y),
                 max_abs_err(h_s_ref, h_s_n, N_st), TOL);
    print_result("VLLM-style kernel",
                 max_abs_err(h_y_ref, h_y_v, N_y),
                 max_abs_err(h_s_ref, h_s_v, N_st), TOL);

    // benchmarking
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      launch_base(
        d_u, d_delta, d_A, d_B, d_C, d_D, d_dbias,
        use_z ? d_z : nullptr, /*delta_softplus=*/true,
        batch, dim, dstate, seqlen, d_states, d_y
      );
    }
    HOST_CALL((void)0);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of base kernel %f (ms)\n", (time * 1e-6f) / repeat);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) {
      int N_carry = batch * dim * dstate * n_chunks;
      Buffer *d_carry_a, *d_carry_b;
      HOST_CALL(buffer_create(&d_carry_a, N_carry * SZ));
      HOST_CALL(buffer_zero(d_carry_a, 0, N_carry * SZ));
      HOST_CALL(buffer_create(&d_carry_b, N_carry * SZ));
      HOST_CALL(buffer_zero(d_carry_b, 0, N_carry * SZ));
      launch_kernel(p, use_z, /*softplus=*/true, even_len, d_carry_a, d_carry_b);
      HOST_CALL((void)0);
      HOST_CALL(buffer_free(d_carry_a));
      HOST_CALL(buffer_free(d_carry_b));
    }
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of vllm-style kernel %f (ms)\n", (time * 1e-6f) / repeat);

    free(h_u); free(h_delta); free(h_A); free(h_B); free(h_C);
    free(h_D); free(h_dbias); free(h_z); free(h_states0);
    free(h_s_ref); free(h_s_n); free(h_s_v);
    free(h_y_ref); free(h_y_n); free(h_y_v);

    HOST_CALL(buffer_free(d_u));
    HOST_CALL(buffer_free(d_delta));
    HOST_CALL(buffer_free(d_A));
    HOST_CALL(buffer_free(d_B));
    HOST_CALL(buffer_free(d_C));
    HOST_CALL(buffer_free(d_D));
    HOST_CALL(buffer_free(d_dbias));
    HOST_CALL(buffer_free(d_z));
    HOST_CALL(buffer_free(d_states));
    HOST_CALL(buffer_free(d_y));
    platform.finalize();
    return 0;
}
