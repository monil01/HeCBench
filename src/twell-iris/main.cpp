// IRIS dmem host for the original TwELL gated FFN benchmark.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <initializer_list>
#include "torch_compat.h"
#include <iris/iris.hpp>

static int irisDeviceIndex=-1;
static void check(int code) {if(code!=IRIS_SUCCESS){fprintf(stderr,"IRIS error %d\n",code);std::exit(2);}}
template<typename T> struct Buffer {
    std::vector<T> host;
    iris::DMem mem;
    explicit Buffer(size_t n):host(n),mem(host.data(),host.size()*sizeof(T)) {}
};
static void launch(const char* name,dim3 grid,dim3 block,
                   std::initializer_list<std::pair<void*,int>> arguments,
                   std::initializer_list<iris::DMem*> outputs={}) {
    std::vector<void*> params;std::vector<int> modes;
    for(auto argument:arguments){params.push_back(argument.first);modes.push_back(argument.second);}
    size_t global[]={size_t(grid.x)*block.x,size_t(grid.y)*block.y,size_t(grid.z)*block.z};
    size_t local[]={block.x,block.y,block.z};
    int dimensions=(grid.z>1||block.z>1)?3:(grid.y>1||block.y>1)?2:1;
    iris::Task task;
    check(task.kernel(name,dimensions,nullptr,global,local,params.size(),params.data(),modes.data()));
    for(auto output:outputs)check(task.flush_out(*output));
    check(task.submit(irisDeviceIndex,nullptr,true));
}
using BFloat=at::BFloat16;
static void fill(Buffer<BFloat>& buffer,unsigned seed,float scale,float offset) {
    size_t n=buffer.host.size();
    launch("fill_bf16_kernel",dim3((n+255)/256),dim3(256),
           {{&buffer.mem,iris_w},{&n,sizeof(n)},{&seed,sizeof(seed)},{&scale,sizeof(scale)},{&offset,sizeof(offset)}});
}
template<typename T> static void dump(const char* suffix,const Buffer<T>& buffer) {
    if(const char* prefix=std::getenv("HECBENCH_DUMP_PREFIX")) {
        std::string filename=std::string(prefix)+suffix;
        FILE* file=fopen(filename.c_str(),"wb");
        if(!file||fwrite(buffer.host.data(),sizeof(T),buffer.host.size(),file)!=buffer.host.size()) {perror("dump");std::exit(3);}
        fclose(file);
    }
}
static void gated(Buffer<BFloat>& a,Buffer<BFloat>& gate,Buffer<BFloat>& up,
                  Buffer<BFloat>& down,Buffer<uint32_t>& packed,Buffer<BFloat>& out,
                  size_t M,int hidden,int intermediate,bool high_precision,bool flush=false) {
    launch("mm_mfma_d2t_kernel",dim3(intermediate/256,M/32),dim3(256),
           {{&a.mem,iris_r},{&gate.mem,iris_r},{&packed.mem,iris_rw},
            {&M,sizeof(M)},{&hidden,sizeof(hidden)},{&intermediate,sizeof(intermediate)}});
    std::string name=std::string("t2d_")+(high_precision?"fp32_":"bf16_")+(intermediate==5632?"22":"32");
    if(flush)
        launch(name.c_str(),dim3(M),dim3(32),
               {{&a.mem,iris_r},{&packed.mem,iris_r},{&up.mem,iris_r},{&down.mem,iris_r},{&out.mem,iris_w}},
               {&out.mem,&packed.mem});
    else
        launch(name.c_str(),dim3(M),dim3(32),
               {{&a.mem,iris_r},{&packed.mem,iris_r},{&up.mem,iris_r},{&down.mem,iris_r},{&out.mem,iris_w}});
}
struct Args {
    int hidden = 2048;
    int intermediate = 8192;
    int batch = 2;
    int seq = 2048;
    int layers = 16;
    int reps = 50;
    int warmup = 10;
    bool high_precision = false;
    float gate_bias = -0.0013f;  // negative bias -> sparse gate activations
};

static int parse_int(const char* v) { return std::atoi(v); }
static float parse_float(const char* v) { return static_cast<float>(std::atof(v)); }

static void parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", k.c_str());
                std::exit(1);
            }
            return argv[++i];
        };
        if (k == "--hidden") a.hidden = parse_int(next());
        else if (k == "--intermediate") a.intermediate = parse_int(next());
        else if (k == "--batch") a.batch = parse_int(next());
        else if (k == "--seq") a.seq = parse_int(next());
        else if (k == "--layers") a.layers = parse_int(next());
        else if (k == "--reps") a.reps = parse_int(next());
        else if (k == "--warmup") a.warmup = parse_int(next());
        else if (k == "--high-precision") a.high_precision = true;
        else if (k == "--gate-bias") a.gate_bias = parse_float(next());
        else if (k == "--help" || k == "-h") {
            printf(
                "Usage: %s [options]\n"
                "  --hidden N         hidden size / OUT_DIM (must be 2048)\n"
                "  --intermediate N   intermediate / FEATURE_DIM (5632 or 8192)\n"
                "  --batch N          batch size (default 2)\n"
                "  --seq N            sequence length (default 2048)\n"
                "  --layers N         gated MLP layers in stack (default 16)\n"
                "  --reps N           timed reps (default 50)\n"
                "  --warmup N         warmup reps (default 10)\n"
                "  --high-precision   use fp32 fused gated path\n"
                "  --gate-bias F      negative GATE bias for sparsity (default -0.0013)\n",
                argv[0]);
            std::exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", k.c_str());
            std::exit(1);
        }
    }
}

static void validate_shape(const Args& a, size_t M) {
    if (a.hidden != 2048) {
        fprintf(stderr, "error: --hidden must be 2048 (got %d)\n", a.hidden);
        std::exit(1);
    }
    if (a.intermediate % 256 != 0 ||
        (a.intermediate / 256 != 22 && a.intermediate / 256 != 32)) {
        fprintf(stderr, "error: --intermediate must be 5632 or 8192 (got %d)\n",
                a.intermediate);
        std::exit(1);
    }
    if (M % 256 != 0) {
        fprintf(stderr, "error: batch*seq must be divisible by 256 (got %zu)\n",
                M);
        std::exit(1);
    }
}

// weight amplitude (uniform half-width) for GATE/UP/DOWN and activations
static constexpr float kWeightScale = 0.10f;   // -> U[-0.05, 0.05]
static constexpr float kActScale = 0.50f;      // activations in [0, 0.5]
static constexpr float kActOffset = 0.25f;


static int run_validate(const Args& args) {
    const size_t M=size_t(args.batch)*args.seq;
    validate_shape(args,M);
    const int hidden=args.hidden,intermediate=args.intermediate;
    printf("Validate: M=%zu hidden=%d intermediate=%d gate_bias=%.5f precision=%s\n",
           M,hidden,intermediate,args.gate_bias,args.high_precision?"fp32":"bf16");
    size_t wN=size_t(intermediate)*hidden,actN=M*hidden;
    Buffer<BFloat> gate(wN),up(wN),down(wN),a(actN),a_twell(actN),ref(actN),out(actN);
    fill(gate,0x11u,kWeightScale,args.gate_bias);
    fill(up,0x22u,kWeightScale,0.0f);fill(down,0x33u,kWeightScale,0.0f);
    fill(a,0x44u,kActScale,kActOffset);
    launch("copy_bf16",dim3((actN+255)/256),dim3(256),
           {{&a.mem,iris_r},{&a_twell.mem,iris_w},{&actN,sizeof(actN)}});
    Buffer<float> H(M*intermediate),gate_raw(M*intermediate),up_raw(M*intermediate);
    Buffer<uint8_t> active(M*intermediate);
    Buffer<uint32_t> packed(M*(intermediate/8));
    int mi=int(M),ni=intermediate,ki=hidden;
    launch("gemm_nt_bf16",dim3(intermediate/128,M/128),dim3(256),
           {{&a.mem,iris_r},{&gate.mem,iris_r},{&gate_raw.mem,iris_w},{&mi,sizeof(mi)},{&ni,sizeof(ni)},{&ki,sizeof(ki)}});
    launch("gemm_nt_bf16",dim3(intermediate/128,M/128),dim3(256),
           {{&a.mem,iris_r},{&up.mem,iris_r},{&up_raw.mem,iris_w},{&mi,sizeof(mi)},{&ni,sizeof(ni)},{&ki,sizeof(ki)}});
    long total=long(M)*intermediate;
    launch("fuse_gate_up_kernel",dim3((total+255)/256),dim3(256),
           {{&gate_raw.mem,iris_r},{&up_raw.mem,iris_r},{&H.mem,iris_w},{&active.mem,iris_w},{&total,sizeof(total)}});
    ni=hidden;ki=intermediate;
    launch("gemm_nn_down",dim3(hidden/128,M/128),dim3(256),
           {{&H.mem,iris_r},{&down.mem,iris_r},{&ref.mem,iris_w},{&mi,sizeof(mi)},{&ni,sizeof(ni)},{&ki,sizeof(ki)}},{&ref.mem});
    const int tiles=intermediate/256;
    Buffer<int> counts(M*tiles);long total_tiles=long(M)*tiles;
    ni=intermediate;
    launch("tile_count_kernel",dim3((total_tiles+255)/256),dim3(256),
           {{&active.mem,iris_r},{&counts.mem,iris_w},{&mi,sizeof(mi)},{&ni,sizeof(ni)}},{&counts.mem});
    long long sum_active=0;int max_tile=0;
    for(int count:counts.host){sum_active+=count;if(count>max_tile)max_tile=count;}
    double density=double(sum_active)/(double(M)*intermediate);
    const int payload_slots=31;bool overflow=max_tile>payload_slots;
    printf("Gate sparsity: density=%.3f%%  avg_nnz/tile=%.1f  max_nnz/tile=%d (budget=%d)\n",
           density*100.0,double(sum_active)/(double(M)*tiles),max_tile,payload_slots);
    if(overflow)printf("WARNING: max_nnz/tile exceeds the %d-slot packed budget; TwELL is\n"
                      "         lossy in this regime. Increase |--gate-bias| for a valid test.\n",payload_slots);
    gated(a_twell,gate,up,down,packed,out,M,hidden,intermediate,args.high_precision,true);
    auto bf16_to_f32=[](BFloat b){uint16_t word;std::memcpy(&word,&b,2);uint32_t bits=uint32_t(word)<<16;float result;std::memcpy(&result,&bits,4);return result;};
    double max_abs=0,sum_abs=0,sum_ref=0,max_ref=0,max_rel=0;
    for(size_t i=0;i<actN;++i) {
        double r=bf16_to_f32(ref.host[i]),t=bf16_to_f32(out.host[i]),d=std::fabs(r-t);
        max_abs=std::max(max_abs,d);sum_abs+=d;sum_ref+=std::fabs(r);max_ref=std::max(max_ref,std::fabs(r));
        max_rel=std::max(max_rel,d/(std::fabs(r)+1e-6));
    }
    double mean_abs=sum_abs/actN,mean_ref=sum_ref/actN;
    printf("\nCorrectness (TwELL %s vs fp32 reference)\n",args.high_precision?"fp32":"bf16");
    printf("  reference   : mean|out|=%.5f  max|out|=%.5f\n",mean_ref,max_ref);
    printf("  abs error   : mean=%.6f  max=%.6f\n",mean_abs,max_abs);
    printf("  rel error   : mean=%.4f%%  max(vs mean|out|)=%.4f%%\n",
           (mean_abs/(mean_ref+1e-12))*100.0,(max_abs/(mean_ref+1e-12))*100.0);
    bool pass=!overflow && mean_abs/(mean_ref+1e-12)<0.03;
    printf("  RESULT      : %s\n",pass?"PASS":"FAIL");
    dump(".validation.ref.bin",ref);dump(".validation.out.bin",out);dump(".validation.packed.bin",packed);
    return pass?0:1;
}
static int run_benchmark(const Args& args) {
    const size_t M=size_t(args.batch)*args.seq;validate_shape(args,M);
    const int hidden=args.hidden,intermediate=args.intermediate;
    printf("Config: tokens(M)=%zu hidden=%d intermediate=%d layers=%d reps=%d warmup=%d precision=%s gate_bias=%.5f\n",
           M,hidden,intermediate,args.layers,args.reps,args.warmup,args.high_precision?"fp32":"bf16",args.gate_bias);
    const size_t wN=size_t(intermediate)*hidden,actN=M*hidden;
    std::vector<std::unique_ptr<Buffer<BFloat>>> gate,up,down;
    for(int l=0;l<args.layers;++l) {
        gate.emplace_back(new Buffer<BFloat>(wN));up.emplace_back(new Buffer<BFloat>(wN));down.emplace_back(new Buffer<BFloat>(wN));
        fill(*gate.back(),0x1000u+l,kWeightScale,args.gate_bias);
        fill(*up.back(),0x2000u+l,kWeightScale,0.0f);fill(*down.back(),0x3000u+l,kWeightScale,0.0f);
    }
    Buffer<BFloat> a(actN),b(actN);fill(a,0xABCDu,kActScale,kActOffset);
    Buffer<uint32_t> packed(M*(intermediate/8));
    auto run_stack=[&](bool flush) {
        Buffer<BFloat>* in=&a;Buffer<BFloat>* out=&b;
        for(int l=0;l<args.layers;++l) {
            gated(*in,*gate[l],*up[l],*down[l],packed,*out,M,hidden,intermediate,args.high_precision,flush && l==args.layers-1);
            std::swap(in,out);
        }
    };
    printf("Warmup (%d reps)...\n",args.warmup);
    for(int r=0;r<args.warmup;++r)run_stack(false);
    printf("Timing (%d reps)...\n",args.reps);
    auto start=std::chrono::steady_clock::now();
    for(int r=0;r<args.reps;++r)run_stack(std::getenv("HECBENCH_DUMP_PREFIX") && r==args.reps-1);
    auto end=std::chrono::steady_clock::now();
    auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
    double total_ms=ns*1e-6,per_rep_ms=total_ms/args.reps,per_layer_ms=per_rep_ms/args.layers;
    double tokens_per_s=(M*1000.0)/per_rep_ms;
    double dense_flops_per_rep=double(M)*intermediate*(6.0*hidden+1.0)*args.layers;
    double tflops=(dense_flops_per_rep/(per_rep_ms/1000.0))/1e12;
    printf("\nResults\n  total time            : %.3f ms (%d reps)\n",total_ms,args.reps);
    printf("  per forward (stack)   : %.4f ms\n  per gated MLP layer   : %.4f ms\n",per_rep_ms,per_layer_ms);
    printf("  input throughput      : %.1f tokens/s\n  dense-equivalent perf : %.2f TFLOP/s\n",tokens_per_s,tflops);
    dump(".benchmark.out.bin",args.layers%2?b:a);
    return 0;
}
int main(int argc,char** argv) {
    Args args;parse_args(argc,argv,args);
    const size_t M=size_t(args.batch)*args.seq;validate_shape(args,M);
    if(M==0||args.layers<=0||args.reps<=0||args.warmup<0){fprintf(stderr,"Invalid nonpositive workload\n");return 1;}
    iris::Platform platform;check(platform.init(nullptr,nullptr,true));
    int backend=-1,type=0;
    for(int d=0;d<platform.ndevs();++d){check(iris_device_info(d,iris_backend,&backend,nullptr));if(backend==iris_hip){irisDeviceIndex=d;break;}}
    if(irisDeviceIndex<0){fprintf(stderr,"No IRIS HIP GPU\n");return 2;}
    char name[256]={};check(iris_device_info(irisDeviceIndex,iris_name,name,nullptr));check(iris_device_info(irisDeviceIndex,iris_type,&type,nullptr));
    int wave=0;
    {iris::DMem info(&wave,sizeof(wave));launch("wave_info",dim3(1),dim3(64),{{&info,iris_w}},{&info});}
    printf("IRIS backend=%d type=%d device=%s wave=%d\n",backend,type,name,wave);
    if(wave!=64){fprintf(stderr,"Preserved native HIP kernel requires wave64\n");return 2;}
    if(run_validate(args)==0)run_benchmark(args);
    check(platform.finalize());return 0;
}
