include(joinpath(@__DIR__,"study_arithmetic.jl"))
function zoom_in!(input,out,H,W,Ho,Wo,pitch,hs,he,ws,we,slots)
 tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1);bx=Int32(blockIdx().x-1);by=Int32(blockIdx().y-1);bz=Int32(blockIdx().z-1)
 s=CUDA.CuDynamicSharedArray(Float32,slots);state=load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch);CUDA.sync_threads();output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,Val(1));return
end
function zoom_out!(input,out,H,W,Ho,Wo,pitch,hs,he,ws,we,slots)
 tx=Int32(threadIdx().x-1);ty=Int32(threadIdx().y-1);bx=Int32(blockIdx().x-1);by=Int32(blockIdx().y-1);bz=Int32(blockIdx().z-1)
 s=CUDA.CuDynamicSharedArray(Float32,slots);state=load_tile!(s,input,tx,ty,bx,by,bz,H,W,Ho,Wo,pitch);CUDA.sync_threads();output_tile!(s,out,tx,ty,H,W,Ho,Wo,hs,he,ws,we,state,Val(2));return
end
function zoom_edge!(out,H,W,pitch,hs,ws,he,we)
 w=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);h=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);b=Int32(blockIdx().z-1);edge!(out,h,w,b,H,W,pitch,hs,ws,he,we);return
end
