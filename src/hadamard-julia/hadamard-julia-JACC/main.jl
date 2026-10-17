using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function hadamard_stage!(index,input,output,dim,stride,scale)
    i=Int32(index)-Int32(1)
    if i<length(input)
        col=i%dim;partner=i ⊻ stride
        @inbounds u=Float32(input[i+Int32(1)]);v=Float32(input[partner+Int32(1)])
        value=(col&stride)==0 ? u+v : v-u
        @inbounds output[i+Int32(1)]=eltype(output)(value*scale)
    end
    nothing
end
function build(stages,dim,n)
    backend=JACC.get_backend(:cuda);spec=JACC.LaunchSpec{typeof(backend)}(;threads=256,blocks=cld(n,256),shmem_size=0,sync=false)
    ()->(for (source,dest,stride,scale) in stages;JACC.parallel_for(hadamard_stage!,spec,n,source,dest,dim,stride,scale);end)
end
function study_prepare(x,out,a,b,dim,scale)
    stages=[];source=x;stride=1;index=0
    while stride<dim
        index+=1;last=stride*2==dim;dest=last ? out : (isodd(index) ? a : b);push!(stages,(source,dest,Int32(stride),last ? scale : 1f0));source=dest;stride*=2
    end
    build(stages,Int32(dim),length(x))
end
exit(study_main(study_prepare))
