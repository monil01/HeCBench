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
#include "study_cases.h"
#define CK(x) do{if((x)!=0)throw std::runtime_error(#x);}while(0)
const int G=16,R=10;const unsigned GUARD=0xa5a55a5a,POISON=0x13579bdf;
std::vector<unsigned>readv(const char*n){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("fixtureopen");size_t z=f.tellg();if(z%4)throw std::runtime_error("dtype");f.seekg(0);std::vector<unsigned>v(z/4);if(!f.read((char*)v.data(),z))throw std::runtime_error("shortfixture");return v;}
unsigned*upload(std::vector<unsigned>&v){unsigned*p;CK(cudaMalloc(&p,v.size()*4));CK(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice));return p;}
std::vector<unsigned>get(unsigned*p,int n){std::vector<unsigned>v(n);CK(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}
void append(std::vector<unsigned>&o,std::vector<unsigned>v){o.insert(o.end(),v.begin(),v.end());}
std::vector<unsigned>initial(int n){std::vector<unsigned>v(n+32,GUARD);std::fill(v.begin()+G,v.end()-G,POISON);return v;}
struct State{Case c;std::vector<unsigned>host[6];unsigned*dev[6];};
struct Capt{std::vector<unsigned>prefix,post;double seconds;bool ok;};
Capt pipeline(State&s,std::vector<unsigned>&oracle){auto c=s.c;Capt out;out.ok=true;cusparseHandle_t h;cusparseSpMatDescr_t a,b,cc;cusparseSpGEMMDescr_t desc;CK(cusparseCreate(&h));cusparsePointerMode_t pm;cudaStream_t stream;CK(cusparseGetPointerMode(h,&pm));CK(cusparseGetStream(h,&stream));if(pm!=CUSPARSE_POINTER_MODE_HOST||stream!=0)throw std::runtime_error("vendorpolicy");
CK(cusparseCreateCsr(&a,c.m,c.k,c.annz,s.dev[1]+G,s.dev[2]+G,s.dev[0]+G,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));CK(cusparseCreateCsr(&b,c.k,c.n,c.bnnz,s.dev[4]+G,s.dev[5]+G,s.dev[3]+G,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));CK(cusparseCreateCsr(&cc,c.m,c.n,0,NULL,NULL,NULL,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));CK(cusparseSpGEMM_createDescr(&desc));auto cohost=initial(c.m+1);unsigned*co=upload(cohost);for(int k=0;k<6;k++)append(out.prefix,get(s.dev[k],s.host[k].size()));append(out.prefix,get(co,cohost.size()));float alpha=1,beta=0;size_t bs1=0,bs2=0;void*buf1,*buf2;
CK(cusparseSpGEMM_workEstimation(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,a,b,&beta,cc,CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs1,NULL));CK(cudaMalloc(&buf1,std::max(bs1,size_t(1))));CK(cusparseSpGEMM_workEstimation(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,a,b,&beta,cc,CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs1,buf1));CK(cusparseSpGEMM_compute(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,a,b,&beta,cc,CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs2,NULL));CK(cudaMalloc(&buf2,std::max(bs2,size_t(1))));CK(cudaDeviceSynchronize());
auto start=std::chrono::steady_clock::now();for(int r=0;r<R;r++)CK(cusparseSpGEMM_compute(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,a,b,&beta,cc,CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,desc,&bs2,buf2));CK(cudaDeviceSynchronize());out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R;
int64_t cm,cn,nnz;CK(cusparseSpMatGetSize(cc,&cm,&cn,&nnz));if(cm!=c.m||cn!=c.n||nnz!=c.cnnz)throw std::runtime_error("Cshape/structuralnnz differsoracle; nooutofbounds copyallowed");auto cvhost=initial(nnz),cihost=initial(nnz);unsigned*cv=upload(cvhost),*ci=upload(cihost);CK(cusparseCsrSetPointers(cc,co+G,ci+G,cv+G));CK(cusparseSpGEMM_copy(h,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,a,b,&beta,cc,CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,desc));CK(cudaDeviceSynchronize());
for(int k=0;k<6;k++){auto v=get(s.dev[k],s.host[k].size());out.ok&=v==s.host[k];append(out.post,v);}int off=c.oo;for(auto spec:std::vector<std::pair<unsigned*,int>>{{cv,c.cnnz},{co,c.m+1},{ci,c.cnnz}}){auto v=get(spec.first,spec.second+32);for(int i=0;i<spec.second+32;i++)out.ok&=v[i]==(i>=G&&i<spec.second+G?oracle[off+i-G]:GUARD);off+=spec.second;append(out.post,v);}
CK(cusparseSpGEMM_destroyDescr(desc));CK(cusparseDestroySpMat(a));CK(cusparseDestroySpMat(b));CK(cusparseDestroySpMat(cc));CK(cusparseDestroy(h));for(auto p:{co,cv,ci})CK(cudaFree(p));CK(cudaFree(buf1));CK(cudaFree(buf2));return out;}
int main(){void*lib=dlopen("/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140",RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());void*fn=dlsym(lib,"cusparseSpGEMM_compute");Dl_info info;if(!fn||fn!=(void*)&cusparseSpGEMM_compute||!dladdr(fn,&info))throw std::runtime_error("functionresolution");cusparseHandle_t h;CK(cusparseCreate(&h));int version;CK(cusparseGetVersion(h,&version));CK(cusparseDestroy(h));printf("STUDY_VENDOR SpGEMM_compute=%s version=%d stream=0 alpha1beta0 DEFAULT F32 NONTRANSPOSE\n",info.dli_fname,version);auto inputs=readv("input_u32.bin"),oracle=readv("oracle_u32.bin");std::vector<State>states;
for(auto c:cases){State s;s.c=c;int sizes[]={c.annz,c.m+1,c.annz,c.bnnz,c.k+1,c.bnnz};int off=c.io;for(int k=0;k<6;k++){s.host[k].assign(sizes[k]+32,GUARD);std::copy(inputs.begin()+off,inputs.begin()+off+sizes[k],s.host[k].begin()+G);off+=sizes[k];s.dev[k]=upload(s.host[k]);}auto warm=pipeline(s,oracle);if(!warm.ok)throw std::runtime_error("warmoracle");states.push_back(s);}
for(int trial=1;trial<=5;trial++){std::vector<unsigned>prefix,post;bool ok=true;double seconds=0;for(auto&s:states){auto c=pipeline(s,oracle);append(prefix,c.prefix);append(post,c.post);seconds+=c.seconds;ok&=c.ok;}append(prefix,post);std::string path="spmm_trial"+std::to_string(trial)+".bin";std::ofstream(path,std::ios::binary).write((char*)prefix.data(),prefix.size()*4);printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_five_fresh_descriptor_SpGEMM_compute_only_batches_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"}}\n",trial,ok?"true":"false",seconds,path.c_str(),prefix.size(),prefix.size());}for(auto&s:states)for(auto p:s.dev)CK(cudaFree(p));}
