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
const G=16;const GUARD=UInt32(0xa5a55a5a);const POISON=UInt32(0xc640e600);const R=10
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140";const LIBSHA="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298";const CASES=[(m=1,n=1,nnz=1,mode=0,io=0,oo=0,max_abs_row_product_units_1_over512=1792,strict_lower_abs_row_sum_upper=0.109375,minimum_diagonal=2,maximum_diagonal=4,),(m=17,n=3,nnz=50,mode=1,io=5,oo=1,max_abs_row_product_units_1_over512=0,strict_lower_abs_row_sum_upper=0.109375,minimum_diagonal=2,maximum_diagonal=4,),(m=33,n=5,nnz=150,mode=2,io=174,oo=52,max_abs_row_product_units_1_over512=1814,strict_lower_abs_row_sum_upper=0.109375,minimum_diagonal=2,maximum_diagonal=4,),(m=257,n=7,nnz=1536,mode=3,io=673,oo=217,max_abs_row_product_units_1_over512=1830,strict_lower_abs_row_sum_upper=0.109375,minimum_diagonal=2,maximum_diagonal=4,),(m=20082,n=128,nnz=150616,mode=4,io=5802,oo=2016,max_abs_row_product_units_1_over512=1835,strict_lower_abs_row_sum_upper=0.109375,minimum_diagonal=2,maximum_diagonal=4,)]
function readwords(n)
 b=read(joinpath(@__DIR__,n));length(b)%4==0||error("fixturedtype");collect(reinterpret(UInt32,b))
end
function capture!(out,s)
 for a in s.dev;append!(out,Array(a));end
end
function run_study()
 bytes2hex(sha256(read(LIBPATH)))==LIBSHA||error("libraryhash");lib=Libdl.dlopen(LIBPATH);sym(n)=Libdl.dlsym(lib,n);ck(x)=x==0||error("cuSPARSE status $x");create=sym(:cusparseCreate);destroy=sym(:cusparseDestroy);csr=sym(:cusparseCreateCsr);dn=sym(:cusparseCreateDnMat);setattr=sym(:cusparseSpMatSetAttribute);cdesc=sym(:cusparseSpSM_createDescr);ddesc=sym(:cusparseSpSM_destroyDescr);dmat=sym(:cusparseDestroySpMat);ddn=sym(:cusparseDestroyDnMat);query=sym(:cusparseSpSM_bufferSize);analysis=sym(:cusparseSpSM_analysis);solve=sym(:cusparseSpSM_solve)
 info=Ref{NTuple{4,Ptr{Cvoid}}}();ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),solve,info)!=0||error("dladdr");resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));resolved==realpath(LIBPATH)||error("functionresolution");println("STUDY_VENDOR SpSM_solve=",resolved," sha256=",LIBSHA)
 # The oracle stays on the CPU. No CuArray is ever constructed from solution values.
 input=readwords("input_u32.bin");oracle=readwords("oracle_f32.bin");alpha=Ref{Cfloat}(1);states=Any[]
 function operation(s)
  GC.@preserve s alpha begin
   ck(ccall(solve,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,Cint,Ptr{Cvoid}),s.h,0,0,alpha,s.a,s.b,s.cc,0,0,s.desc))
  end
 end
 for c in CASES
  sizes=[c.nnz,c.m+1,c.nnz,c.m*c.n,c.m*c.n];host=Vector{UInt32}[];dev=CuArray{UInt32,1}[];off=c.io
  for k in 1:5
   values=k<=4 ? input[off+1:off+sizes[k]] : fill(POISON,sizes[k]);k<=4&&(off+=sizes[k]);a=vcat(fill(GUARD,G),values,fill(GUARD,G));push!(host,a);push!(dev,CuArray(a))
  end
  h=Ref{Ptr{Cvoid}}();ck(ccall(create,Cint,(Ref{Ptr{Cvoid}},),h));version=Ref{Cint}();pm=Ref{UInt32}();stream=Ref{Ptr{Cvoid}}();ck(ccall(sym(:cusparseGetVersion),Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));ck(ccall(sym(:cusparseGetPointerMode),Cint,(Ptr{Cvoid},Ref{UInt32}),h[],pm));ck(ccall(sym(:cusparseGetStream),Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));version[]==12001&&pm[]==0&&stream[]==C_NULL||error("vendorpolicy")
  a=Ref{Ptr{Cvoid}}();b=Ref{Ptr{Cvoid}}();cc=Ref{Ptr{Cvoid}}();desc=Ref{Ptr{Cvoid}}();bytes=Ref{Csize_t}()
  GC.@preserve dev alpha begin
   ck(ccall(csr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),a,c.m,c.m,c.nnz,pointer(dev[2],G+1),pointer(dev[3],G+1),pointer(dev[1],G+1),2,2,0,0))
   ck(ccall(dn,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),b,c.m,c.n,c.n,pointer(dev[4],G+1),0,2));ck(ccall(dn,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),cc,c.m,c.n,c.n,pointer(dev[5],G+1),0,2));ck(ccall(cdesc,Cint,(Ref{Ptr{Cvoid}},),desc));lower=Ref{UInt32}(0);nonunit=Ref{UInt32}(0);ck(ccall(setattr,Cint,(Ptr{Cvoid},Cint,Ref{UInt32},Csize_t),a[],0,lower,4));ck(ccall(setattr,Cint,(Ptr{Cvoid},Cint,Ref{UInt32},Csize_t),a[],1,nonunit,4));ck(ccall(query,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,Cint,Ptr{Cvoid},Ref{Csize_t}),h[],0,0,alpha,a[],b[],cc[],0,0,desc[],bytes))
   buf=CuArray{UInt8}(undef,max(Int(bytes[]),1));CUDA.device_synchronize()
   GC.@preserve buf begin
    ck(ccall(analysis,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,Cint,Ptr{Cvoid},CUDA.CuPtr{Cvoid}),h[],0,0,alpha,a[],b[],cc[],0,0,desc[],pointer(buf)))
   end
   CUDA.device_synchronize();s=(c=c,host=host,dev=dev,h=h[],a=a[],b=b[],cc=cc[],desc=desc[],buf=buf);operation(s);CUDA.device_synchronize();warm=Array(dev[5]);expected=vcat(fill(GUARD,G),oracle[c.oo+1:c.oo+c.m*c.n],fill(GUARD,G))
   if warm!=expected
    diagnostic=UInt32[];capture!(diagnostic,s);path=joinpath("/home/imo/HeCBench_julia/.porting-state/portable_julia/spsm_reference_review","julia_warm_m$(c.m)_pid$(getpid()).bin");open(path,"w")do io;write(io,diagnostic);end;error("warmoracle rawcapture=$path")
   end
   println("STUDY_VENDOR case_m=",c.m," rhs=",c.n," nnz=",c.nnz," version=",version[]," workspace_bytes=",bytes[]," LOWER NON_UNIT ROW alpha1");push!(states,s)
  end
 end
 for trial in 1:5
  prefix=UInt32[];post=UInt32[];ok=true;seconds=0.
  for s in states
   copyto!(s.dev[5],s.host[5]);CUDA.device_synchronize();capture!(prefix,s);start=time_ns();for _ in 1:R;operation(s);end;CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9/R
   for k in 1:5
    actual=Array(s.dev[k]);expected=k<=4 ? s.host[k] : vcat(fill(GUARD,G),oracle[s.c.oo+1:s.c.oo+s.c.m*s.c.n],fill(GUARD,G));ok&=actual==expected;append!(post,actual)
   end
  end
  append!(prefix,post);path="spsm_trial$trial.bin";open(path,"w")do io;write(io,prefix);end
  @printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":10,\"timing_region\":\"sum_five_analyzed_lower_nonunit_SpSM_batches_completion_per_repeat\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),seconds,path,length(prefix),length(prefix))
 end
 for s in states
  ck(ccall(dmat,Cint,(Ptr{Cvoid},),s.a));ck(ccall(ddn,Cint,(Ptr{Cvoid},),s.b));ck(ccall(ddn,Cint,(Ptr{Cvoid},),s.cc));ck(ccall(ddesc,Cint,(Ptr{Cvoid},),s.desc));ck(ccall(destroy,Cint,(Ptr{Cvoid},),s.h))
 end
end
