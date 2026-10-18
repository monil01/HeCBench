include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
function forward_jacc!(g,pref,act,n::Int32)
 i=Int32(g-1)
 if i<n;@inbounds act[i+1]=forward_value(pref[i+1],i);end
 return
end
function backward_jacc!(g,act,gin,gout,n::Int32)
 i=Int32(g-1)
 if i<n;@inbounds gout[i+1]=backward_value(act[i+1],gin[i+1],i);end
 return
end
function update_jacc!(g,weights,gout,n::Int32)
 i=Int32(g-1)
 if i<n;@inbounds weights[i+1]=muladd(-0.01f0,gout[i+1],weights[i+1]);end
 return
end
function prefetch_jacc!(g,pref,n::Int32,batch::Int32)
 i=Int32(g-1)
 if i<n;@inbounds pref[i+1]=prefetch_value(i,batch);end
 return
end
