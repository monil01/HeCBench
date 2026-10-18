using Libdl
const LIB=Libdl.dlopen(joinpath(pwd(),"libstudy.so"))
const RUN=Libdl.dlsym(LIB,:study_run)
struct PointStruct
 weight::Float32
 pad1::UInt32
 assign::Int64
 cost::Float32
 pad2::UInt32
end
sizeof(PointStruct)==24||error("PointABI")
function wrapped_inputs(pp,cp,wp,tp,sp,n,d,K)
 P=unsafe_wrap(CuArray,CuPtr{PointStruct}(pp),(Int(n),);own=false)
 C=unsafe_wrap(CuArray,CuPtr{Float32}(cp),(Int(n)*Int(d),);own=false)
 W=unsafe_wrap(CuArray,CuPtr{Float32}(wp),(Int(n)*(Int(K)+1),);own=false)
 T=unsafe_wrap(CuArray,CuPtr{Int32}(tp),(Int(n),);own=false)
 S=unsafe_wrap(CuArray,CuPtr{UInt8}(sp),(Int(n),);own=false)
 P,C,W,T,S
end
function main()
 CUDA.functional()||error("CUDAnotfunctional")
 cb=@cfunction(callback,Cvoid,(Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid},Cint,Cint,Clong,Cint))
 path=joinpath(@__DIR__,"input_f32.bin");dir=ENV["STUDY_OUTPUT_DIR"]
 ccall(RUN,Cvoid,(Ptr{Cvoid},Cint,Cstring,Cstring),cb,0,path,dir)
 for trial=1:5
  ccall(RUN,Cvoid,(Ptr{Cvoid},Cint,Cstring,Cstring),cb,trial,path,dir)
 end
end
