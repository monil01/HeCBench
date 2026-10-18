using KernelAbstractions
include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
@kernel function forward_ka!(pref,act,n::Int32)
 g=@index(Global,Linear)
 i=Int32(g-1)
 if i<n;@inbounds act[i+1]=forward_value(pref[i+1],i);end
end
@kernel function backward_ka!(act,gin,gout,n::Int32)
 g=@index(Global,Linear)
 i=Int32(g-1)
 if i<n;@inbounds gout[i+1]=backward_value(act[i+1],gin[i+1],i);end
end
@kernel function update_ka!(weights,gout,n::Int32)
 g=@index(Global,Linear)
 i=Int32(g-1)
 if i<n;@inbounds weights[i+1]=muladd(-0.01f0,gout[i+1],weights[i+1]);end
end
@kernel function prefetch_ka!(pref,n::Int32,batch::Int32)
 g=@index(Global,Linear)
 i=Int32(g-1)
 if i<n;@inbounds pref[i+1]=prefetch_value(i,batch);end
end
