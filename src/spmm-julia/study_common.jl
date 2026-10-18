#=
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

=#

using SHA,Libdl,Printf
const G=16;const GUARD=UInt32(0xa5a55a5a);const POISON=UInt32(0x13579bdf);const R=10
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBSHA="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298";const CASES=[(m=1,k=1,n=1,mode=0,annz=1,bnnz=1,cnnz=1,io=0,oo=0,max_absolute_sum_units_1_over32=1,),(m=17,k=11,n=23,mode=1,annz=52,bnnz=57,cnnz=182,io=8,oo=4,max_absolute_sum_units_1_over32=98,),(m=33,k=17,n=19,mode=2,annz=49,bnnz=33,cnnz=80,io=256,oo=386,max_absolute_sum_units_1_over32=333,),(m=64,k=33,n=65,mode=3,annz=486,bnnz=450,cnnz=2383,io=472,oo=580,max_absolute_sum_units_1_over32=135,),(m=1024,k=1024,n=1024,mode=4,annz=1024,bnnz=1024,cnnz=1024,io=2443,oo=5411,max_absolute_sum_units_1_over32=49,)]
function readwords(n)
 b=read(joinpath(@__DIR__,n));length(b)%4==0||error("fixturedtype");collect(reinterpret(UInt32,b))
end
initial(n)=vcat(fill(GUARD,G),fill(POISON,n),fill(GUARD,G))
function run_study()
 bytes2hex(sha256(read(LIBPATH)))==LIBSHA||error("libraryhash");lib=Libdl.dlopen(LIBPATH);sym(n)=Libdl.dlsym(lib,n);ck(x)=x==0||error("cuSPARSE status $x");create=sym(:cusparseCreate);destroy=sym(:cusparseDestroy);csr=sym(:cusparseCreateCsr);dmat=sym(:cusparseDestroySpMat);cdesc=sym(:cusparseSpGEMM_createDescr);ddesc=sym(:cusparseSpGEMM_destroyDescr);estimate=sym(:cusparseSpGEMM_workEstimation);compute=sym(:cusparseSpGEMM_compute);copyfn=sym(:cusparseSpGEMM_copy);getsize=sym(:cusparseSpMatGetSize);setptr=sym(:cusparseCsrSetPointers)
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),compute,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==realpath(LIBPATH)||error("functionresolution");println("STUDY_VENDOR SpGEMM_compute=",resolved," sha256=",LIBSHA)
 input=readwords("input_u32.bin");oracle=readwords("oracle_u32.bin");states=Any[]
 # The same signature is used for workEstimation and compute, including size and external workspace.
 function phase(fn,h,alpha,beta,a,b,c,desc,bytes,buffer)
  ck(ccall(fn,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,Ptr{Cvoid},Ref{Csize_t},CUDA.CuPtr{Cvoid}),h,0,0,alpha,a,b,beta,c,0,0,desc,bytes,buffer))
 end
 function pipeline(s)
  c=s.c;h=Ref{Ptr{Cvoid}}();ck(ccall(create,Cint,(Ref{Ptr{Cvoid}},),h));version=Ref{Cint}();pm=Ref{UInt32}();stream=Ref{Ptr{Cvoid}}();ck(ccall(sym(:cusparseGetVersion),Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));ck(ccall(sym(:cusparseGetPointerMode),Cint,(Ptr{Cvoid},Ref{UInt32}),h[],pm));ck(ccall(sym(:cusparseGetStream),Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));version[]==12001&&pm[]==0&&stream[]==C_NULL||error("vendorpolicy")
  co=CuArray(initial(c.m+1));owners=Any[s,co];prefix=UInt32[];post=UInt32[];ok=true;seconds=0.
  GC.@preserve owners begin
   a=Ref{Ptr{Cvoid}}();b=Ref{Ptr{Cvoid}}();cc=Ref{Ptr{Cvoid}}();desc=Ref{Ptr{Cvoid}}();null=CUDA.CuPtr{Cvoid}(0)
   for (target,rows,cols,nnz,base) in [(a,c.m,c.k,c.annz,0),(b,c.k,c.n,c.bnnz,3)]
    ck(ccall(csr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),target,rows,cols,nnz,pointer(s.dev[base+2],G+1),pointer(s.dev[base+3],G+1),pointer(s.dev[base+1],G+1),2,2,0,0))
   end
   ck(ccall(csr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),cc,c.m,c.n,0,null,null,null,2,2,0,0));ck(ccall(cdesc,Cint,(Ref{Ptr{Cvoid}},),desc));for dev in s.dev;append!(prefix,Array(dev));end;append!(prefix,Array(co));alpha=Ref{Cfloat}(1);beta=Ref{Cfloat}(0);bs1=Ref{Csize_t}(0);bs2=Ref{Csize_t}(0)
   phase(estimate,h[],alpha,beta,a[],b[],cc[],desc[],bs1,null);buf1=CuArray{UInt8}(undef,max(Int(bs1[]),1));push!(owners,buf1);phase(estimate,h[],alpha,beta,a[],b[],cc[],desc[],bs1,pointer(buf1));phase(compute,h[],alpha,beta,a[],b[],cc[],desc[],bs2,null);buf2=CuArray{UInt8}(undef,max(Int(bs2[]),1));push!(owners,buf2);CUDA.device_synchronize();start=time_ns()
   for _ in 1:R;phase(compute,h[],alpha,beta,a[],b[],cc[],desc[],bs2,pointer(buf2));end
   CUDA.device_synchronize();seconds=(time_ns()-start)*1e-9/R
   cm=Ref{Int64}();cn=Ref{Int64}();nnz=Ref{Int64}();ck(ccall(getsize,Cint,(Ptr{Cvoid},Ref{Int64},Ref{Int64},Ref{Int64}),cc[],cm,cn,nnz));cm[]==c.m&&cn[]==c.n&&nnz[]==c.cnnz||error("Cshape/structuralnnz differsoracle; nooutofbounds copyallowed");cv=CuArray(initial(c.cnnz));ci=CuArray(initial(c.cnnz));push!(owners,cv,ci);ck(ccall(setptr,Cint,(Ptr{Cvoid},CUDA.CuPtr{Cint},CUDA.CuPtr{Cint},CUDA.CuPtr{Cvoid}),cc[],reinterpret(CUDA.CuPtr{Cint},pointer(co,G+1)),reinterpret(CUDA.CuPtr{Cint},pointer(ci,G+1)),pointer(cv,G+1)));CUDA.device_synchronize()
   ck(ccall(copyfn,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,Ptr{Cvoid}),h[],0,0,alpha,a[],b[],beta,cc[],0,0,desc[]));CUDA.device_synchronize()
   for (dev,host) in zip(s.dev,s.host);actual=Array(dev);ok&=actual==host;append!(post,actual);end;off=c.oo
   for (dev,n) in [(cv,c.cnnz),(co,c.m+1),(ci,c.cnnz)]
    actual=Array(dev);expected=vcat(fill(GUARD,G),oracle[off+1:off+n],fill(GUARD,G));ok&=actual==expected;append!(post,actual);off+=n
   end
   ck(ccall(ddesc,Cint,(Ptr{Cvoid},),desc[]));for mat in [a[],b[],cc[]];ck(ccall(dmat,Cint,(Ptr{Cvoid},),mat));end;ck(ccall(destroy,Cint,(Ptr{Cvoid},),h[]))
  end
  (prefix=prefix,post=post,seconds=seconds,ok=ok)
 end
 for c in CASES
  sizes=[c.annz,c.m+1,c.annz,c.bnnz,c.k+1,c.bnnz];host=Vector{UInt32}[];dev=CuArray{UInt32,1}[];off=c.io
  for n in sizes
   a=vcat(fill(GUARD,G),input[off+1:off+n],fill(GUARD,G));off+=n;push!(host,a);push!(dev,CuArray(a))
  end
  s=(c=c,host=host,dev=dev);CUDA.device_synchronize();warm=pipeline(s);warm.ok||error("warmoracle");push!(states,s)
 end
 for trial in 1:5
  prefix=UInt32[];post=UInt32[];ok=true;seconds=0.
  for s in states
   c=pipeline(s);append!(prefix,c.prefix);append!(post,c.post);seconds+=c.seconds;ok&=c.ok
  end
  append!(prefix,post);path="spmm_trial$trial.bin";open(path,"w")do io;write(io,prefix);end
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_five_fresh_descriptor_SpGEMM_compute_only_batches_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),seconds,path,length(prefix),length(prefix))
 end
end
