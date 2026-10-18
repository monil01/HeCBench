#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include "../hexciton-cuda/utils.hpp"
#include "../hexciton-cuda/kernels.cu"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>std::vector<T>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<T>v(n/sizeof(T));f.read((char*)v.data(),n);return v;}
constexpr float SENT=-8192.25f;
struct Case{int n,mode,kid,reps;std::vector<float>hh,sh,zero;float*h,*s,*out;};
void launch(Case&c){int kernel_id=c.kid,dim=7,num=c.n;real_2_t*d_sigma_in=(real_2_t*)(c.s+4);real_2_t*d_sigma_out=(real_2_t*)(c.out+4);real_2_t*d_hamiltonian=(real_2_t*)(c.h+4);
switch(kernel_id) {
      case 0:  {
        dim3 k0_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k0_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_empty<<<k0_gws, k0_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // initial kernel
      case 1: {
        dim3 k1_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k1_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_init<<<k1_gws, k1_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // refactored initial kernel
      case 2: {
        dim3 k2_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k2_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_refactor<<<k2_gws, k2_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // refactored initial kernel with direct store
      case 3: {
        dim3 k3_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k3_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_refactor_direct_store<<<k3_gws, k3_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range
      case 4: {
        dim3 k4_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k4_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_aosoa_naive<<<k4_gws, k4_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range and compile time constants
      case 5: {
        dim3 k5_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k5_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_aosoa_naive_constants<<<k5_gws, k5_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range, compile time constants, and permuted loops with temporaries
      case 6: {
        dim3 k6_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k6_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_aosoa_naive_constants_perm<<<k6_gws, k6_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 1D range and direct store
      case 7: {
        dim3 k7_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k7_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_aosoa_naive_direct<<<k7_gws, k7_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 1D range, compile time constants, and direct store
      case 8: {
        dim3 k8_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k8_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_aosoa_naive_constants_direct<<<k8_gws, k8_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 1D range, compile time constants, direct store, and permuted loops with temporaries
      case 9: {
        dim3 k9_gws (num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG));
        dim3 k9_lws (VEC_LENGTH_AUTO * PACKAGES_PER_WG);
        comm_aosoa_naive_constants_direct_perm<<<k9_gws, k9_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 2D-range
      case 10: {
        dim3 k10_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k10_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        comm_aosoa<<<k10_gws, k10_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 2D-range and compile-time constants
      case 11: {
        dim3 k11_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k11_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        comm_aosoa_constants<<<k11_gws, k11_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 2D-range, compile-time constants, and permuted loops with temporaries
      case 12: {
        dim3 k12_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k12_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        comm_aosoa_constants_perm<<<k12_gws, k12_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with 2D range and direct store
      case 13: {
        dim3 k13_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k13_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        comm_aosoa_direct<<<k13_gws, k13_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // vectorised kernel with 2D range, compile-time constants, and direct store
      case 14: {
        dim3 k14_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k14_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        comm_aosoa_constants_direct<<<k14_gws, k14_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // vectorised kernel with compile-time constants, direct store, and permuted loops with temporaries
      case 15: {
        dim3 k15_gws (1, num / VEC_LENGTH_AUTO / PACKAGES_PER_WG);
        dim3 k15_lws (VEC_LENGTH_AUTO, PACKAGES_PER_WG);
        comm_aosoa_constants_direct_perm<<<k15_gws, k15_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel
      case 16: {
        dim3 k16_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k16_lws (VEC_LENGTH);
        comm_manual_aosoa<<<k16_gws, k16_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // manually vectorised kernel with compile-time constants
      case 17: {
        dim3 k17_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k17_lws (VEC_LENGTH);
        comm_manual_aosoa_constants<<<k17_gws, k17_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with compile-time constants and permuted loops with temporaries
      case 18: {
        dim3 k18_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k18_lws (VEC_LENGTH);
        comm_manual_aosoa_constants_perm<<<k18_gws, k18_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with compile-time constants and prefetch
      case 19: {
        dim3 k19_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k19_lws (VEC_LENGTH);
        comm_manual_aosoa_constants_perm_prefetch<<<k19_gws, k19_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with direct store
      case 20: {
        dim3 k20_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k20_lws (VEC_LENGTH);
        comm_manual_aosoa_direct<<<k20_gws, k20_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, dim);
        break;
      }

      // manually vectorised kernel with compile time constants and direct store
      case 21: {
        dim3 k21_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k21_lws (VEC_LENGTH);
        comm_manual_aosoa_constants_direct<<<k21_gws, k21_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // manually vectorised kernel with compile time constants, direct store, and prefetch
      case 22: {
        dim3 k22_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k22_lws (VEC_LENGTH);
        comm_manual_aosoa_constants_direct_prefetch<<<k22_gws, k22_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

       // manually vectorised kernel with compile time constants, direct store, and permuted loops with temporaries
      case 23: {
        dim3 k23_gws (num / (VEC_LENGTH * VEC_LENGTH));
        dim3 k23_lws (VEC_LENGTH);
        comm_manual_aosoa_constants_direct_perm<<<k23_gws, k23_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian);
        break;
      }

      // final GPGPU kernel optimised for an Nvidia GPU
      case 24: {
        size_t block_dim_x = (dim * dim + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        size_t block_dim_y = NUM_SUB_GROUPS;
        dim3 k24_gws (num / (block_dim_y * CHUNK_SIZE), 1);
        dim3 k24_lws (block_dim_x, block_dim_y);
        final_gpu_kernel<<<k24_gws, k24_lws>>>(d_sigma_in, d_sigma_out, d_hamiltonian, num);
        break;
      }
      default: std::cerr << "ERROR: **** benchmark kernel unavailable **** \n";
    }
}
int main(){auto f=load<float>(HX_FIXTURE);auto o=load<double>(HX_ORACLE);std::vector<Case>cs;size_t i=0;while(i<f.size()){Case c;c.n=f[i];c.mode=f[i+1];c.kid=f[i+2];c.reps=f[i+4];if(f[i+5]!=hdt){fprintf(stderr,"SourceHDTbitsMismatch\n");return 3;}i+=6;int hn=98,sn=98*c.n;c.hh.assign(hn+8,SENT);std::copy(f.begin()+i,f.begin()+i+hn,c.hh.begin()+4);i+=hn;c.sh.assign(sn+8,SENT);std::copy(f.begin()+i,f.begin()+i+sn,c.sh.begin()+4);i+=sn;c.zero.assign(sn,0.f);ck(cudaMalloc(&c.h,c.hh.size()*4));ck(cudaMalloc(&c.s,c.sh.size()*4));ck(cudaMalloc(&c.out,(sn+8)*4));ck(cudaMemcpy(c.h,c.hh.data(),c.hh.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(c.s,c.sh.data(),c.sh.size()*4,cudaMemcpyHostToDevice));cs.push_back(c);}
 for(auto&c:cs){ck(cudaMemcpy(c.out+4,c.zero.data(),c.zero.size()*4,cudaMemcpyHostToDevice));launch(c);ck(cudaDeviceSynchronize());}
 for(int trial=1;trial<=5;trial++){auto payload=f;bool ok=true;double tm=0,err=0;size_t oi=0;std::vector<double>times;for(auto&c:cs){int sn=98*c.n;std::vector<float>got(sn+8,SENT);ck(cudaMemcpy(c.out,got.data(),got.size()*4,cudaMemcpyHostToDevice));double ct=0;for(int rep=0;rep<c.reps;rep++){ck(cudaMemcpy(c.out+4,c.zero.data(),c.zero.size()*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();launch(c);ck(cudaDeviceSynchronize());ct+=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count();}ct/=c.reps;tm+=ct;times.push_back(ct);std::vector<float>hb(c.hh.size()),sb(c.sh.size());ck(cudaMemcpy(hb.data(),c.h,hb.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(sb.data(),c.s,sb.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(got.data(),c.out,got.size()*4,cudaMemcpyDeviceToHost));ok&=memcmp(hb.data(),c.hh.data(),hb.size()*4)==0&&memcmp(sb.data(),c.sh.data(),sb.size()*4)==0;for(int j=0;j<4;j++)ok&=got[j]==SENT;for(int j=sn+4;j<sn+8;j++)ok&=got[j]==SENT;payload.insert(payload.end(),hb.begin(),hb.end());payload.insert(payload.end(),sb.begin(),sb.end());if(c.kid>0){for(int j=4;j<sn+4;j++){double e=fabs(double(got[j])-o[oi]);err=fmax(err,e);ok&=std::isfinite(got[j])&&e<=1e-6+1e-4*fabs(o[oi++]);if(c.mode==2)ok&=got[j]==0.f;}payload.insert(payload.end(),got.begin(),got.end());}else{payload.insert(payload.end(),got.begin(),got.begin()+4);payload.insert(payload.end(),got.end()-4,got.end());}}
 std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream z(path,std::ios::binary);z.write((char*)payload.data(),payload.size()*4);z.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum75_original_1001_repeat_kernel_means_complete\",\"jit_excluded\":true,\"internal_repeats\":1001,\"timed_kernel_calls\":75075,\"numeric_cases\":72,\"execution_only_cases\":3,\"case_times_seconds\":[",trial,tm,ok?"true":"false",payload.size(),err);for(size_t j=0;j<times.size();j++)printf("%s%.17g",j?",":"",times[j]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok)return 3;}}
