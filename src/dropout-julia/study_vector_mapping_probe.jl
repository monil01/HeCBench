# Untimed emitted-code inspection only: two vector launches, no benchmark timing series.
using CUDA
model=ARGS[1];directory=abspath(ARGS[2]);mkpath(directory)
suffix=model=="cuda_jl" ? "cuda.jl" : model=="ka" ? "KA" : "JACC"
path=joinpath(@__DIR__,"dropout-julia-"*suffix,"main.jl")
source=read(path,String);stop=findfirst("include(joinpath(@__DIR__,\"..\",\"study_common.jl\"))",source)
stop===nothing&&error("MissingHarnessBoundary")
include_string(Main,source[1:prevind(source,first(stop))],path)
allocate=model=="ka" ? ((T,n)->KernelAbstractions.allocate(BACKEND,T,n)) : model=="jacc" ? ((T,n)->JACC.zeros(T,n)) : ((T,n)->CUDA.zeros(T,n))
for vec in (2,4)
 n=1028;a=allocate(Float32,n+8);b=allocate(Float32,n+8);m=allocate(UInt8,n+32);fill!(a,.1f0)
 VF=NTuple{vec,VecElement{Float32}};VM=NTuple{vec,VecElement{UInt8}}
 av=reinterpret(VF,view(a,5:n+4));bv=reinterpret(VF,view(b,5:n+4));mv=reinterpret(VM,view(m,17:n+16))
 filename=joinpath(directory,model*"_vec"*string(vec)*".ptx")
 open(filename,"w")do io
  CUDA.@device_code_ptx io=io launch(vec,av,bv,mv,UInt32(n),.5f0,UInt64(12345678),UInt64(87654321))
  CUDA.synchronize(CUDA.context())
 end
 println("MAPPING_PTX model=",model," vec=",vec," payload_bytes=",sizeof(VF)," alignment=",Base.datatype_alignment(VF)," raw_bytes=",filesize(filename))
end
