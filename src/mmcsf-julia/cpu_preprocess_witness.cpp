#include <fstream>
#include <cstdio>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <cmath> 
#include <bits/stdc++.h> 
#include "configured/mttkrp_cpu.h"
#include "configured/mttkrp_gpu.h" 


int main(int argc, char* argv[]){ 

  Options Opt = parse_cmd_options(argc, argv);

  printf("\nStarting MTTKRP..\n");

  Tensor X;
  load_tensor(X, Opt);

  auto t0 = std::chrono::steady_clock::now();
  sort_COOtensor(X);
  auto t1 = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  printf("Sort time : %.3f sec \n", time * 1e-9);

  check_opt(X, Opt); //check options are good

  Matrix *U = new Matrix[X.ndims]; 
  create_mats(X, U, Opt, false);
  randomize_mats(X, U, Opt);

  // Collect slice and fiber stats: Create CSF for all modes
  int redunMode = -1;

  Tensor *arrX = new Tensor[X.ndims]; 


  for (int m = 0; m < X.ndims; ++m){

    init_tensor(arrX, X, Opt, m);

    if(m == 1){
      if (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && 
          arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1]){
        redunMode = m;
      }
    }
    else if(m == 2){
      if ((arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]) || 
          (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])){
        redunMode = m;
      }
    }

    else if(m == 3){
      if ((arrX[m].modeOrder[1] == arrX[m-3].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-3].modeOrder[1]) ||
          (arrX[m].modeOrder[1] == arrX[m-2].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-2].modeOrder[1]) ||
          (arrX[m].modeOrder[1] == arrX[m-1].modeOrder[0] && 
           arrX[m].modeOrder[0] == arrX[m-1].modeOrder[1])){
        redunMode = m;
      }
    }

    if(m !=  redunMode) sort_COOtensor(arrX[m]);

    if(m !=  redunMode) create_HCSR(arrX[m], Opt); 
  }       

  Opt.redunMode = redunMode;

  TiledTensor ModeWiseTiledX[X.ndims];
  t0 = std::chrono::steady_clock::now();
  mm_partition_reuseBased(arrX, X, ModeWiseTiledX, Opt);
  populate_paritions(X, ModeWiseTiledX);
  t1 = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  printf("mm_partition & populate - time: %.3f sec \n", time * 1e-9);

  t0 = std::chrono::steady_clock::now();
  for (int m = 0; m < X.ndims; ++m){

    if(ModeWiseTiledX[m].totNnz > 0){           
      sort_MI_CSF(X, ModeWiseTiledX, m);
      create_TiledHCSR(ModeWiseTiledX, Opt, m);
      create_fbrLikeSlcInds(ModeWiseTiledX, m);
    }
  }
  t1 = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  printf("Sort,createCSF,createFbrIND - time: %.3f sec \n", time * 1e-9);

  size_t total=0; for(size_t m=0;m<X.ndims;m++) total+=ModeWiseTiledX[m].totNnz;
  if(total!=X.totNnz) return 7;
  printf("CPU_PREPROCESS_COMPLETE dims=%zu nnz=%zu partition_nnz=%zu\n",X.ndims,X.totNnz,total);
  return 0;
}
