using CUDA,KernelAbstractions
const MODEL="ka"
include("../study_device.jl")
@kernel cpu=false function interval_kernel!(out,counts,domain,choice)
 tid=Int32(@index(Global,Linear))-Int32(1)
 interval_body!(tid,out,counts,domain,choice)
end
@kernel cpu=false function echo_kernel!(out,domain)
 tid=Int32(@index(Global,Linear))-Int32(1)
 interval_echo_body!(tid,out,domain)
end
function launch_main(s,choice)
 interval_kernel!(CUDABackend(),64)(s.result.data,s.counts.data,s.domain,Int32(choice);ndrange=65536);nothing
end
function launch_echo(s)
 echo_kernel!(CUDABackend(),64)(s.echo.data,s.domain;ndrange=65536);nothing
end
include("../study_common.jl")
main()
