include(joinpath(@__DIR__,"..","study_portable_arithmetic.jl"))
function up1!(g,x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 flat=UInt64(g-1)
 if flat<B*C*H*W
 bc=div(flat,H*W);i=mod(div(flat,W),H);j=mod(flat,W);tile!(x,out,H,W,bc,i,j,Val(false))
 end
 return
end
function up2!(gx,gy,gz,x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 j=UInt64(gx-1);i=UInt64(gy-1);bc=UInt64(gz-1)
 if i<H&&j<W;tile!(x,out,H,W,bc,i,j,Val(false));end
 return
end
function up3!(g,x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 flat=UInt64(g-1)
 if flat<B*C*H*W
 bc=div(flat,H*W);i=mod(div(flat,W),H);j=mod(flat,W);tile!(x,out,H,W,bc,i,j,Val(true))
 end
 return
end
function up4!(gx,gy,gz,x,out,B::UInt64,C::UInt64,H::UInt64,W::UInt64)
 j=UInt64(gx-1);i=UInt64(gy-1);bc=UInt64(gz-1)
 if i<H&&j<W;tile!(x,out,H,W,bc,i,j,Val(true));end
 return
end
