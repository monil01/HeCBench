using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function hadamard_stage!(input,output,dim,stride,scale)
    i=Int32(@index(Global,Linear))-Int32(1)
    if i<length(input)
        col=i%dim;partner=i ⊻ stride
        @inbounds u=Float32(input[i+Int32(1)]);v=Float32(input[partner+Int32(1)])
        value=(col&stride)==0 ? u+v : v-u
        @inbounds output[i+Int32(1)]=eltype(output)(value*scale)
    end
    nothing
end
function build(stages,dim,n)
    k=hadamard_stage!(CUDABackend(),256)
    ()->(for (source,dest,stride,scale) in stages;k(source,dest,dim,stride,scale;ndrange=n);end)
end
function study_prepare(x,out,a,b,dim,scale)
    stages=[];source=x;stride=1;index=0
    while stride<dim
        index+=1;last=stride*2==dim;dest=last ? out : (isodd(index) ? a : b);push!(stages,(source,dest,Int32(stride),last ? scale : 1f0));source=dest;stride*=2
    end
    build(stages,Int32(dim),length(x))
end
exit(study_main(study_prepare))
