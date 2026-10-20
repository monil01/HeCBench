using CUDA,KernelAbstractions
const MODEL="ka"
# Explicit CUDA-backend hybrid: own KA entry dispatch, source warp collectives/shared helpers.
include("../study_device.jl")
@kernel function small_entry!(idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 rsmt_body!(Val(24),Val(64),Val(false),idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
end
@kernel function large_entry!(idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 rsmt_body!(Val(3),Val(512),Val(true),idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
end
function launch!(c,blocks)
 small_entry!(get_backend(c.x.d),768)(c.idx.d,c.xin.d,c.yin.d,c.oi.d,c.x.d,c.y.d,c.es,c.ed,c.wl.d,c.state.d,Int32(c.nets);ndrange=blocks*768)
 large_entry!(get_backend(c.x.d),96)(c.idx.d,c.xin.d,c.yin.d,c.oi.d,c.x.d,c.y.d,c.es,c.ed,c.wl.d,c.state.d,Int32(c.nets);ndrange=blocks*96)
 nothing
end
include("../study_common.jl")
study()
