using KernelAbstractions
include(joinpath(@__DIR__,"..","study_portable_arithmetic.jl"))
@kernel function up1!(x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 g=@index(Global,Linear)
 flat=UInt64(g-1)
 if flat<B*C*H*W
 bc=div(flat,H*W);i=mod(div(flat,W),H);j=mod(flat,W);tile!(x,out,H,W,bc,i,j,Val(false))
 end
end
@kernel function up2!(x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 g=@index(Global,Cartesian)
 j=UInt64(g[1]-1);i=UInt64(g[2]-1);bc=UInt64(g[3]-1)
 if i<H&&j<W;tile!(x,out,H,W,bc,i,j,Val(false));end
end
@kernel function up3!(x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 g=@index(Global,Linear)
 flat=UInt64(g-1)
 if flat<B*C*H*W
 bc=div(flat,H*W);i=mod(div(flat,W),H);j=mod(flat,W);tile!(x,out,H,W,bc,i,j,Val(true))
 end
end
@kernel function up4!(x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 g=@index(Global,Cartesian)
 j=UInt64(g[1]-1);i=UInt64(g[2]-1);bc=UInt64(g[3]-1)
 if i<H&&j<W;tile!(x,out,H,W,bc,i,j,Val(true));end
end
