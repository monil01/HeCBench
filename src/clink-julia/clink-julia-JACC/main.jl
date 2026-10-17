using JACC
include("../study_common.jl")
@inline function kernel(i,x,w,y)
 if i<=512;lstm_lane!(x,w,y,i);end
 return
end
const BACKEND=JACC.get_backend(:cuda)
const SPEC=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=2,shmem_size=0,sync=false)
clink_study((x,w,y)->()->JACC.parallel_for(SPEC,512,kernel,x,w,y))
