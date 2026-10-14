#include <iris/iris.hpp>
#include <vector>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
using FLOAT=float;
const int NTHR_PER_BLK = 256;           // Number of CUDA threads per block
const int NBLOCK  = 56*4;               // Number of CUDA blocks (SMs on P100)
const int Npoint = NBLOCK*NTHR_PER_BLK; // No. of independent samples
const int Neq = 100000;                 // No. of generations to equilibrate
const int Ngen_per_block = 5000;        // No. of generations per block
const float DELTA = 2.0;                // Random step size

// Explicitly typed constants so can easily work with both floats and floats
static const FLOAT FOUR = 4.0;
static const FLOAT TWO  = 2.0;
static const FLOAT ONE  = 1.0;
static const FLOAT HALF = 0.5;
static const FLOAT ZERO = 0.0;


void vmc_task(const char* name,int blocks,int threads,std::vector<void*> args,std::vector<int> info) {
 size_t global[]={size_t(blocks)*threads},local[]={size_t(threads)};
 iris::Task task; task.kernel(name,1,nullptr,global,local,args.size(),args.data(),info.data());
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) throw std::runtime_error(name);
}
void sum_stats(const int n,iris::DMem& stats,iris::DMem& statsum,iris::DMem& blocksums) {
 for(int what=0;what<4;what++) {
  int input_offset=what*n,output_offset=0;
  vmc_task("SumWithinBlocks",NBLOCK,NTHR_PER_BLK,{(void*)&n,(void*)&stats,(void*)&blocksums,(void*)&input_offset,(void*)&output_offset},{sizeof(n),iris_r,iris_w,sizeof(input_offset),sizeof(output_offset)});
  input_offset=0;output_offset=what;
  vmc_task("SumWithinBlocks",1,NBLOCK,{(void*)&NBLOCK,(void*)&blocksums,(void*)&statsum,(void*)&input_offset,(void*)&output_offset},{sizeof(NBLOCK),iris_r,iris_rw,sizeof(input_offset),sizeof(output_offset)});
 }
}
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <number of blocks to sample>\n", argv[0]);
    return 1;
  }
  const int Nsample = atoi(argv[1]); // No. of blocks to sample

  iris::Platform platform;platform.init(&argc,&argv,1);
  std::vector<FLOAT> h_x1(Npoint); iris::DMem x1(h_x1.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_y1(Npoint); iris::DMem y1(h_y1.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_z1(Npoint); iris::DMem z1(h_z1.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_x2(Npoint); iris::DMem x2(h_x2.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_y2(Npoint); iris::DMem y2(h_y2.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_z2(Npoint); iris::DMem z2(h_z2.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_psi(Npoint); iris::DMem psi(h_psi.data(),Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_stats(4*Npoint); iris::DMem stats(h_stats.data(),4*Npoint*sizeof(FLOAT));
  std::vector<FLOAT> h_statsum(4); iris::DMem statsum(h_statsum.data(),4*sizeof(FLOAT));
  std::vector<FLOAT> h_blocksums(NBLOCK); iris::DMem blocksums(h_blocksums.data(),NBLOCK*sizeof(FLOAT));
  std::vector<unsigned int> h_ranstates(Npoint); iris::DMem ranstates(h_ranstates.data(),Npoint*sizeof(unsigned int));
  int seed=5551212;

  vmc_task("initran",NBLOCK,NTHR_PER_BLK,{(void*)&seed,(void*)&ranstates},{sizeof(seed),iris_w});
  vmc_task("initialize",NBLOCK,NTHR_PER_BLK,{(void*)&x1,(void*)&y1,(void*)&z1,(void*)&x2,(void*)&y2,(void*)&z2,(void*)&psi,(void*)&ranstates},{iris_w,iris_w,iris_w,iris_w,iris_w,iris_w,iris_w,iris_rw});
  vmc_task("zero_stats",NBLOCK,NTHR_PER_BLK,{(void*)&Npoint,(void*)&stats},{sizeof(Npoint),iris_w});

  // Equilibrate
  vmc_task("propagate",NBLOCK,NTHR_PER_BLK,{(void*)&Npoint,(void*)&Neq,(void*)&x1,(void*)&y1,(void*)&z1,(void*)&x2,(void*)&y2,(void*)&z2,(void*)&psi,(void*)&stats,(void*)&ranstates},{sizeof(Npoint),sizeof(Neq),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw});

  // Accumulators for averages over blocks --- use doubles
  double r1_tot = ZERO,  r1_sq_tot = ZERO;
  double r2_tot = ZERO,  r2_sq_tot = ZERO;
  double r12_tot = ZERO, r12_sq_tot = ZERO;
  double naccept = ZERO;  // Keeps track of propagation efficiency

  double time = 0.0;

  for (int sample=0; sample<Nsample; sample++) {


    auto start = std::chrono::steady_clock::now();

    vmc_task("zero_stats",NBLOCK,NTHR_PER_BLK,{(void*)&Npoint,(void*)&stats},{sizeof(Npoint),iris_w});
    vmc_task("propagate",NBLOCK,NTHR_PER_BLK,{(void*)&Npoint,(void*)&Ngen_per_block,(void*)&x1,(void*)&y1,(void*)&z1,(void*)&x2,(void*)&y2,(void*)&z2,(void*)&psi,(void*)&stats,(void*)&ranstates},{sizeof(Npoint),sizeof(Ngen_per_block),iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw});

    struct {FLOAT r1, r2, r12, accept;} s;
    sum_stats(Npoint, stats, statsum, blocksums);


    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    iris::Task flush; flush.flush_out(statsum);flush.submit(iris_gpu,nullptr,1);
    memcpy(&s,h_statsum.data(),sizeof(s));

    naccept += s.accept;
    s.r1 /= Ngen_per_block*Npoint;
    s.r2 /= Ngen_per_block*Npoint;
    s.r12 /= Ngen_per_block*Npoint;

#ifdef DEBUG
    printf(" block %6d  %.6f  %.6f  %.6f\n", sample, s.r1, s.r2, s.r12);
#endif

    r1_tot += s.r1;   r1_sq_tot += s.r1*s.r1;
    r2_tot += s.r2;   r2_sq_tot += s.r2*s.r2;
    r12_tot += s.r12; r12_sq_tot += s.r12*s.r12;
  }

  r1_tot /= Nsample; r1_sq_tot /= Nsample;
  r2_tot /= Nsample; r2_sq_tot /= Nsample;
  r12_tot /= Nsample; r12_sq_tot /= Nsample;

  double r1s = sqrt((r1_sq_tot - r1_tot*r1_tot) / Nsample);
  double r2s = sqrt((r2_sq_tot - r2_tot*r2_tot) / Nsample);
  double r12s = sqrt((r12_sq_tot - r12_tot*r12_tot) / Nsample);

  printf(" <r1>  = %.6f +- %.6f\n", r1_tot, r1s);
  printf(" <r2>  = %.6f +- %.6f\n", r2_tot, r2s);
  printf(" <r12> = %.6f +- %.6f\n", r12_tot, r12s);

  printf(" acceptance ratio=%.1f%%\n",
         100.0*naccept/double(Npoint)/double(Ngen_per_block)/double(Nsample)); // avoid int overflow

  printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / Nsample);












  platform.finalize();
  return 0;
}
