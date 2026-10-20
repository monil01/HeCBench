using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
# Own JACC entries; CUDA-backend source warp/shared helpers disclosed as hybrid.
include("../study_device.jl")
function small_entry!(index,idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 rsmt_body!(Val(24),Val(64),Val(false),idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 nothing
end
function large_entry!(index,idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 rsmt_body!(Val(3),Val(512),Val(true),idx,xin,yin,oi,x,y,es,ed,wl,state,nets)
 nothing
end
function launch!(c,blocks)
 # Source helper uses static shared storage; reserve no unnecessary dynamic shared bytes.
 a=JACC.launch_spec(threads=768,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(a,blocks*768,small_entry!,c.idx.d,c.xin.d,c.yin.d,c.oi.d,c.x.d,c.y.d,c.es,c.ed,c.wl.d,c.state.d,Int32(c.nets))
 b=JACC.launch_spec(threads=96,blocks=blocks,shmem_size=0,sync=false)
 JACC.parallel_for(b,blocks*96,large_entry!,c.idx.d,c.xin.d,c.yin.d,c.oi.d,c.x.d,c.y.d,c.es,c.ed,c.wl.d,c.state.d,Int32(c.nets))
 nothing
end
include("../study_common.jl")
study()
