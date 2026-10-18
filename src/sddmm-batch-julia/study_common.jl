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
const LIBPATH="/usr/lib/x86_64-linux-gnu/libcusparse.so.12.0.1.140"
const LIBSHA="d4d0efdb21bcc1bd85ae42bec5744350207b22042ab62bc072d355b80e0d5298"
function guarded(::Type{T},values) where T
 host=vcat(fill(T(16384),8),T.(values),fill(T(16384),8));base=CuArray(host)
 (host=host,base=base,n=length(values))
end
function sddmm_study()
 open(LIBPATH) do io;@assert bytes2hex(sha256(io))==LIBSHA;end
 lib=Libdl.dlopen(LIBPATH);sym(n)=Libdl.dlsym(lib,n);ck(x)=x==0||error("cuSPARSE status $x")
 create=sym(:cusparseCreate);dense=sym(:cusparseCreateDnMat);dbatch=sym(:cusparseDnMatSetStridedBatch);csr=sym(:cusparseCreateCsr);cbatch=sym(:cusparseCsrSetStridedBatch);query=sym(:cusparseSDDMM_bufferSize);preprocess=sym(:cusparseSDDMM_preprocess);compute=sym(:cusparseSDDMM)
 info=Ref{NTuple{4,Ptr{Cvoid}}}();@assert ccall(:dladdr,Cint,(Ptr{Cvoid},Ref{NTuple{4,Ptr{Cvoid}}}),compute,info)!=0
 resolved=realpath(unsafe_string(Ptr{UInt8}(info[][1])));@assert resolved==realpath(LIBPATH);println("STUDY_VENDOR SDDMM_dladdr=",resolved," sha256=",LIBSHA)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(pre)==78637&&pre[1]==5&&length(gold)==81442
 cases=Any[];pos=2
 for ci in 1:5
  b,m,k,n,nnz,repeat=Int.(pre[pos:pos+5]);pos+=6
  ns=[b*m*k,b*k*n,b*(m+1),b*nnz,b*nnz];arrays=Any[]
  for (j,N) in enumerate(ns)
   T=j in (3,4) ? Int32 : Float32;push!(arrays,guarded(T,pre[pos:pos+N-1]));pos+=N
  end
  A,B,off,col,C=arrays;h=Ref{Ptr{Cvoid}}();ck(ccall(create,Cint,(Ref{Ptr{Cvoid}},),h));version=Ref{Cint}();pm=Ref{Cint}();stream=Ref{Ptr{Cvoid}}()
  ck(ccall(sym(:cusparseGetVersion),Cint,(Ptr{Cvoid},Ref{Cint}),h[],version));ck(ccall(sym(:cusparseGetPointerMode),Cint,(Ptr{Cvoid},Ref{Cint}),h[],pm));ck(ccall(sym(:cusparseGetStream),Cint,(Ptr{Cvoid},Ref{Ptr{Cvoid}}),h[],stream));@assert version[]==12001&&pm[]==0&&stream[]==C_NULL
  println("STUDY_VENDOR cuSPARSE=",version[]," pointermode=",pm[]," stream=0 batches=",b)
  a=Ref{Ptr{Cvoid}}();bb=Ref{Ptr{Cvoid}}();c=Ref{Ptr{Cvoid}}()
  GC.@preserve arrays begin
   ck(ccall(dense,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),a,m,k,k,pointer(A.base,9),0,2));ck(ccall(dbatch,Cint,(Ptr{Cvoid},Cint,Int64),a[],b,m*k))
   ck(ccall(dense,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},Cint,Cint),bb,k,n,n,pointer(B.base,9),0,2));ck(ccall(dbatch,Cint,(Ptr{Cvoid},Cint,Int64),bb[],b,k*n))
   ck(ccall(csr,Cint,(Ref{Ptr{Cvoid}},Int64,Int64,Int64,CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},CUDA.CuPtr{Cvoid},Cint,Cint,Cint,Cint),c,m,n,nnz,pointer(off.base,9),pointer(col.base,9),pointer(C.base,9),2,2,0,0))
   ck(ccall(cbatch,Cint,(Ptr{Cvoid},Cint,Int64,Int64),c[],b,m+1,nnz))
   alpha=Ref{Cfloat}(1);beta=Ref{Cfloat}(0);bytes=Ref{Csize_t}(0)
   ck(ccall(query,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,Ref{Csize_t}),h[],0,0,alpha,a[],bb[],beta,c[],0,0,bytes))
   buffer=CuArray{UInt8}(undef,max(1,Int(bytes[])))
   ck(ccall(preprocess,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,CUDA.CuPtr{Cvoid}),h[],0,0,alpha,a[],bb[],beta,c[],0,0,pointer(buffer)))
   CUDA.device_synchronize();push!(cases,(b=b,m=m,k=k,n=n,nnz=nnz,repeat=repeat,arrays=arrays,h=h[],a=a[],bb=bb[],c=c[],alpha=alpha,beta=beta,buffer=buffer,workspace_bytes=Int(bytes[])))
  end
 end
 @assert pos==length(pre)+1
 function execute(s)
  A,B,off,col,C=s.arrays;copyto!(C.base,C.host);before=Array(C.base);ok=before==C.host;CUDA.device_synchronize()
  GC.@preserve s begin
   start=time_ns()
   for rep in 1:s.repeat
    ck(ccall(compute,Cint,(Ptr{Cvoid},Cint,Cint,Ref{Cfloat},Ptr{Cvoid},Ptr{Cvoid},Ref{Cfloat},Ptr{Cvoid},Cint,Cint,CUDA.CuPtr{Cvoid}),s.h,0,0,s.alpha,s.a,s.bb,s.beta,s.c,0,0,pointer(s.buffer)))
   end
   CUDA.device_synchronize();seconds=(time_ns()-start)*1e-9
   actual=map(q->Array(q.base),s.arrays);vals=Float64.(actual[5][9:8+C.n])
   for i in 1:4;ok &=actual[i]==s.arrays[i].host;append!(vals,actual[i][9:8+s.arrays[i].n]);end
   append!(vals,before[9:8+C.n])
   for a in actual;guards=vcat(a[1:8],a[end-7:end]);ok &=all(==(16384),guards);append!(vals,guards);end
   (vals=vals,seconds=seconds,ok=ok)
  end
 end
 for c in cases;@assert execute(c).ok;end
 for trial in 1:5
  art=copy(pre);times=Float64[];ok=true
  for c in cases;z=execute(c);append!(art,z.vals);push!(times,z.seconds);ok &=z.ok;end
  ok &=art[length(pre)+1:end]==gold
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum5_strided_batch_SDDMM_vendor_compute_5repeats_completion\",\"jit_excluded\":true,\"internal_repeats\":5,\"authored_device_kernel_entrypoints\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s,\"workspace_bytes\":%s}\n",trial,sum(times),ok ? "true" : "false",length(art),path,length(art),string(times),string([c.workspace_bytes for c in cases]))
 end
 for c in cases
  for mat in (c.a,c.bb);ck(ccall(sym(:cusparseDestroyDnMat),Cint,(Ptr{Cvoid},),mat));end
  ck(ccall(sym(:cusparseDestroySpMat),Cint,(Ptr{Cvoid},),c.c));ck(ccall(sym(:cusparseDestroy),Cint,(Ptr{Cvoid},),c.h))
 end
end
