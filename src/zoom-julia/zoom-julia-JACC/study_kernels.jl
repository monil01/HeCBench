include(joinpath(@__DIR__,"..","study_arithmetic.jl"))
function zoom_in_jacc!(gx,gy,gz,input,out,H,W,Ho,Wo,pitch,hs,he,ws,we,template)
 tx=Int32(mod(gx-1,16));ty=Int32(mod(gy-1,16));bx=Int32(div(gx-1,16));by=Int32(div(gy-1,16));bz=Int32(gz-1)
 s=JACC.shared(template);state=load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch);JACC.sync_workgroup();output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,Val(1));return
end
function zoom_out_jacc!(gx,gy,gz,input,out,H,W,Ho,Wo,pitch,hs,he,ws,we,template)
 tx=Int32(mod(gx-1,16));ty=Int32(mod(gy-1,16));bx=Int32(div(gx-1,16));by=Int32(div(gy-1,16));bz=Int32(gz-1)
 s=JACC.shared(template);state=load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch);JACC.sync_workgroup();output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,Val(2));return
end
function zoom_edge_jacc!(gx,gy,gz,out,H,W,pitch,hs,ws,he,we)
 edge!(out,Int32(gy-1),Int32(gx-1),Int32(gz-1),H,W,pitch,hs,ws,he,we);return
end
