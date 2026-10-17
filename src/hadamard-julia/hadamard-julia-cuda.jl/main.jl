using CUDA
function hadamard_stage!(input,output,dim,stride,scale)
    i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)-Int32(1)
    if i<length(input)
        col=i%dim;partner=i ⊻ stride
        @inbounds u=Float32(input[i+Int32(1)]);v=Float32(input[partner+Int32(1)])
        value=(col&stride)==0 ? u+v : v-u
        @inbounds output[i+Int32(1)]=eltype(output)(value*scale)
    end
    nothing
end
function build(stages,dim,n)
    ()->(for (source,dest,stride,scale) in stages;@cuda threads=256 blocks=cld(n,256) hadamard_stage!(source,dest,dim,stride,scale);end)
end
function study_prepare(x,out,a,b,dim,scale)
    stages=[];source=x;stride=1;index=0
    while stride<dim
        index+=1;last=stride*2==dim;dest=last ? out : (isodd(index) ? a : b);push!(stages,(source,dest,Int32(stride),last ? scale : 1f0));source=dest;stride*=2
    end
    build(stages,Int32(dim),length(x))
end
if abspath(PROGRAM_FILE)==@__FILE__
    include(joinpath(@__DIR__,"..","study_common.jl"));exit(study_main(study_prepare))
end
