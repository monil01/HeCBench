using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function interval_kernel!(out,counts,domain,choice)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(64)+CUDA.threadIdx().x-Int32(1)
 interval_body!(tid,out,counts,domain,choice);nothing
end
function echo_kernel!(out,domain)
 tid=(CUDA.blockIdx().x-Int32(1))*Int32(64)+CUDA.threadIdx().x-Int32(1)
 interval_echo_body!(tid,out,domain);nothing
end
function launch_main(s,choice)
 CUDA.@cuda threads=64 blocks=1024 interval_kernel!(s.result.data,s.counts.data,s.domain,Int32(choice));nothing
end
function launch_echo(s)
 CUDA.@cuda threads=64 blocks=1024 echo_kernel!(s.echo.data,s.domain);nothing
end
include("../study_common.jl")
main()
