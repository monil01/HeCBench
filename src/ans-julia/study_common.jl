using Libdl
const LIB=Libdl.dlopen(joinpath(pwd(),"libstudy.so"));const RUN=Libdl.dlsym(LIB,:study_run)
function callback(op,ptrs,ints)
 p=unsafe_load.(Ref(ptrs),1:6);a=unsafe_load.(Ref(ints),1:9)
 function wrap(i,T,n)
  unsafe_wrap(CuArray,CuPtr{T}(UInt(p[i])),(Int(n),);own=false)
 end
 args=(wrap(1,UInt32,a[7]),wrap(2,UInt32,1024),wrap(3,SyncPoint,a[6]),wrap(4,UInt32,a[6]),wrap(5,UInt8,a[2]),wrap(6,UInt8,a[5]),UInt32(a[1]),UInt32(a[2]),UInt32(a[3]),UInt32(a[4]),UInt32(a[5]))
 launch(Int(op),args,a)
 nothing
end
function main()
 CUDA.functional()||error("CUDA unavailable")
 cb=@cfunction(callback,Cvoid,(Cint,Ptr{Ptr{Cvoid}},Ptr{Cint}))
 input=joinpath(@__DIR__,"input_u32.bin");oracle=joinpath(@__DIR__,"oracle_u8.bin");dir=ENV["STUDY_OUTPUT_DIR"]
 for trial=0:5
  ccall(RUN,Cvoid,(Ptr{Cvoid},Cint,Cstring,Cstring,Cstring),cb,trial,input,oracle,dir)
 end
end
