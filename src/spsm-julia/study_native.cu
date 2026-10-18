/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <dlfcn.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include "study_cases.h"
#define CK(x) do{if((x)!=0)throw std::runtime_error(#x);}while(0)
const int G=16,R=10;const unsigned GUARD=0xa5a55a5a,POISON=0xc640e600;
std::vector<unsigned>readv(const char*n){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixtureopen");size_t z=f.tellg();if(z%4)throw std::runtime_error("dtype");f.seekg(0);std::vector<unsigned>v(z/4);if(!f.read((char*)v.data(),z))throw std::runtime_error("shortfixture");return v;}
unsigned*upload(std::vector<unsigned>&v){unsigned*p;CK(cudaMalloc(&p,v.size()*4));CK(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));return p;}
std::vector<unsigned>get(unsigned*p,int n){std::vector<unsigned>v(n);CK(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
void append(std::vector<unsigned>&o,std::vector<unsigned>v){o.insert(o.end(),v.begin(),v.end());}
struct State{Case c;std::vector<unsigned>host[5];unsigned*dev[5];cusparseHandle_t h;cusparseSpMatDescr_t a;cusparseDnMatDescr_t b,cc;cusparseSpSMDescr_t desc;void*buf;};
void capture(State&s,std::vector<unsigned>&out){for(int k=0;k<5;k++)append(out,get(s.dev[k],s.host[k].size()));}
int main(){void*lib=dlopen("/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());auto solve=(decltype(&cusparseSpSM_solve))dlsym(lib,"cusparseSpSM_solve");Dl_info info;if(!solve||!dladdr((void*)solve,&info))throw std::runtime_error("functionresolution");printf("STUDY_VENDOR SpSM_solve=%s\n",info.dli_fname);auto inputs=readv("input_u32.bin"),oracle=readv("oracle_f32.bin");std::vector<State>states;float alpha=1;
for(auto c:cases){State s;s.c=c;int sizes[]={c.nnz,c.m+1,c.nnz,c.m*c.n,c.m*c.n};int off=c.io;for(int k=0;k<5;k++){s.host[k].assign(sizes[k]+32,GUARD);if(k<4){std::copy(inputs.begin()+off,inputs.begin()+off+sizes[k],s.host[k].begin()+G);off+=sizes[k];}else std::fill(s.host[k].begin()+G,s.host[k].end()-G,POISON);s.dev[k]=upload(s.host[k]);}CK(cusparseCreate(&s.h));int version;cusparsePointerMode_t pm;cudaStream_t stream;CK(cusparseGetVersion(s.h,&version));CK(cusparseGetPointerMode(s.h,&pm));CK(cusparseGetStream(s.h,&stream));if(version!=12001||pm!=CUSPARSE_POINTER_MODE_HOST||stream!=0)throw std::runtime_error("vendorpolicy");CK(cusparseCreateCsr(&s.a,c.m,c.m,c.nnz,s.dev[1]+G,s.dev[2]+G,s.dev[0]+G,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));CK(cusparseCreateDnMat(&s.b,c.m,c.n,c.n,s.dev[3]+G,CUDA_R_32F,CUSPARSE_ORDER_ROW));CK(cusparseCreateDnMat(&s.cc,c.m,c.n,c.n,s.dev[4]+G,CUDA_R_32F,CUSPARSE_ORDER_ROW));CK(cusparseSpSM_createDescr(&s.desc));cusparseFillMode_t fill=CUSPARSE_FILL_MODE_LOWER;cusparseDiagType_t diag=CUSPARSE_DIAG_TYPE_NON_UNIT;CK(cusparseSpMatSetAttribute(s.a,CUSPARSE_SPMAT_FILL_MODE,&fill,sizeof(fill)));CK(cusparseSpMatSetAttribute(s.a,CUSPARSE_SPMAT_DIAG_TYPE,&diag,sizeof(diag)));size_t bytes;CK(cusparseSpSM_bufferSize(s.h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s.a,s.b,s.cc,CUDA_R_32F,CUSPARSE_SPSM_ALG_DEFAULT,s.desc,&bytes));CK(cudaMalloc(&s.buf,std::max(bytes,size_t(1))));CK(cusparseSpSM_analysis(s.h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s.a,s.b,s.cc,CUDA_R_32F,CUSPARSE_SPSM_ALG_DEFAULT,s.desc,s.buf));CK(cudaDeviceSynchronize());CK(solve(s.h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s.a,s.b,s.cc,CUDA_R_32F,CUSPARSE_SPSM_ALG_DEFAULT,s.desc));CK(cudaDeviceSynchronize());auto warm=get(s.dev[4],s.host[4].size());for(int j=0;j<c.m*c.n;j++)if(warm[j+G]!=oracle[c.oo+j]){std::vector<unsigned>diagnostic;capture(s,diagnostic);std::string fail="/home/imo/HeCBench_julia/.porting-state/portable_julia/spsm_reference_review/native_warm_m"+std::to_string(c.m)+"_pid"+std::to_string(getpid())+".bin";std::ofstream(fail,std::ios::binary).write((char*)diagnostic.data(),diagnostic.size()*4);fprintf(stderr,"WARM_MISMATCH m=%d rhs=%d index=%d actualbits=%u expectedbits=%u capture=%s\n",c.m,c.n,j,warm[j+G],oracle[c.oo+j],fail.c_str());throw std::runtime_error("warmoracle");}printf("STUDY_VENDOR case_m=%d rhs=%d nnz=%d version=%d workspace_bytes=%zu LOWER NON_UNIT ROW alpha1\n",c.m,c.n,c.nnz,version,bytes);states.push_back(s);}
for(int trial=1;trial<=5;trial++){std::vector<unsigned>prefix,post;bool ok=true;double seconds=0;for(auto&s:states){auto c=s.c;CK(cudaMemcpy(s.dev[4],s.host[4].data(),s.host[4].size()*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());capture(s,prefix);auto start=std::chrono::steady_clock::now();for(int r=0;r<R;r++)CK(solve(s.h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,s.a,s.b,s.cc,CUDA_R_32F,CUSPARSE_SPSM_ALG_DEFAULT,s.desc));CK(cudaDeviceSynchronize());seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R;for(int k=0;k<5;k++){auto v=get(s.dev[k],s.host[k].size());for(int j=0;j<(int)v.size();j++)ok&=v[j]==(k<4?s.host[k][j]:(j>=G&&j<(int)v.size()-G?oracle[c.oo+j-G]:GUARD));append(post,v);}}append(prefix,post);std::string path="spsm_trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)prefix.data(),prefix.size()*4);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_five_analyzed_lower_nonunit_SpSM_batches_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",seconds,path.c_str(),prefix.size(),prefix.size());}
for(auto&s:states){CK(cusparseDestroySpMat(s.a));CK(cusparseDestroyDnMat(s.b));CK(cusparseDestroyDnMat(s.cc));CK(cusparseSpSM_destroyDescr(s.desc));CK(cusparseDestroy(s.h));CK(cudaFree(s.buf));for(auto p:s.dev)CK(cudaFree(p));}}
