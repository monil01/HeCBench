using CUDA
const MODEL="cuda_jl"
include("../study_device.jl")
function small_entry!(idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 rsmt_body!(Val(24),Val(64),Val(false),idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 nothing
end
function large_entry!(idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 rsmt_body!(Val(3),Val(512),Val(true),idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 nothing
end
function launch!(c,blocks)
 CUDA.@cuda threads=768 blocks=blocks small_entry!(c.idx.d,c.xin.d,c.yin.d,c.oi.d,c.x.d,c.y.d,c.es,c.ed,c.wl.d,c.state.d,Int32(c.nets))
 CUDA.@cuda threads=96 blocks=blocks large_entry!(c.idx.d,c.xin.d,c.yin.d,c.oi.d,c.x.d,c.y.d,c.es,c.ed,c.wl.d,c.state.d,Int32(c.nets))
 nothing
end
include("../study_common.jl")
study()
