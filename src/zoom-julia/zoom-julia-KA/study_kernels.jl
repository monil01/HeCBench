using KernelAbstractions
include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
@kernel function zoom_in_ka!(input,out,H,W,Ho,Wo,pitch,hs,he,ws,we,::Val{SLOTS}) where {SLOTS}
 lc=@index(Local,Cartesian)
 bc=@index(Group,Cartesian)
 tx=Int32(lc[1]-1);ty=Int32(lc[2]-1);bx=Int32(bc[1]-1);by=Int32(bc[2]-1);bz=Int32(bc[3]-1)
 s=@localmem Float32 (SLOTS,)
 state=load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch)
 @synchronize
 output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,Val(1))
end
@kernel function zoom_out_ka!(input,out,H,W,Ho,Wo,pitch,hs,he,ws,we,::Val{SLOTS}) where {SLOTS}
 lc=@index(Local,Cartesian)
 bc=@index(Group,Cartesian)
 tx=Int32(lc[1]-1);ty=Int32(lc[2]-1);bx=Int32(bc[1]-1);by=Int32(bc[2]-1);bz=Int32(bc[3]-1)
 s=@localmem Float32 (SLOTS,)
 state=load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch)
 @synchronize
 output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,Val(2))
end
@kernel function zoom_edge_ka!(out,H,W,pitch,hs,ws,he,we)
 g=@index(Global,Cartesian)
 w=Int32(g[1]-1);h=Int32(g[2]-1);b=Int32(g[3]-1)
 edge!(out,h,w,b,H,W,pitch,hs,ws,he,we)
end
