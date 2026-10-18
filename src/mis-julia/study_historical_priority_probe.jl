using CUDA
path=joinpath(@__DIR__,"mis-julia-cuda.jl","original_main.jl")
source=read(path,String);source=replace(source,"exit(main())"=>"")
include_string(Main,source,path)
nidx=CuArray(Int32[0,1,3,4]);status=CUDA.zeros(UInt8,3)
try
 CUDA.@cuda threads=256 blocks=24 init_kernel!(Int32(3),Int32(4),nidx,status)
 CUDA.synchronize(CUDA.context());println("HISTORICAL_PRIORITY_PROBE_STATUS ",Array(status))
catch err
 showerror(stdout,err,catch_backtrace());println();exit(3)
end
