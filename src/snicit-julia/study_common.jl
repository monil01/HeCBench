using Libdl
const LIB=Libdl.dlopen(joinpath(pwd(),"libstudy.so"));const RUN=Libdl.dlsym(LIB,:study_run)
const STUDY_STREAM=Ref{CUDA.CuStream}()
function callback(op,ptrs,ints,sp)
 p=unsafe_load.(Ref(ptrs),1:13);a=unsafe_load.(Ref(ints),1:8)
 M,N,K,C=Int32.(a[1:4]);fallback=p[1]
 function wrap(i,T,n)
  q=p[i]==C_NULL ? fallback : p[i]
  if p[i]==C_NULL;n=1;end
  unsafe_wrap(CuArray,CuPtr{T}(UInt(q)),(max(Int(n),1),);own=false)
 end
 # Only source-read extents are indexed. Sparse nnz extent derives from actual managed rowpointer.
 nnz=(op==1||op==6) ? Int(unsafe_load(Ptr{Int32}(p[5]),Int(N)+1)) : 1
 args=(wrap(1,Float32,M*(op==3 ? K : N)),wrap(2,Float32,N*K),wrap(3,Float32,K),wrap(4,Float32,M*K),wrap(5,Int32,N+1),wrap(6,Int32,nnz),wrap(7,Float32,nnz),wrap(8,Int32,M),wrap(9,Int32,M),wrap(10,Bool,M),wrap(11,Int32,128),wrap(12,Int32,M),wrap(13,Int32,1),M,N,K,C)
 stream=STUDY_STREAM[]
 launch(Int(op),args,a,stream)
 nothing
end
function main()
 CUDA.functional()||error("CUDA unavailable")
 STUDY_STREAM[]=CUDA.CuStream()
 cb=@cfunction(callback,Cvoid,(Cint,Ptr{Ptr{Cvoid}},Ptr{Cint},Ptr{Cvoid}))
 input=joinpath(@__DIR__,"input_u32.bin");dir=ENV["STUDY_OUTPUT_DIR"]
 for trial=0:5
  ccall(RUN,Cvoid,(Ptr{Cvoid},Cint,Cstring,Cstring,Ptr{Cvoid}),cb,trial,input,dir,STUDY_STREAM[].handle)
 end
end
