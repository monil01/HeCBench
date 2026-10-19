using CUDA
import JACC
JACC.@init_backend
include("../study_device.jl")
function bicluster_kernel!(i,args...)
 x=CUDA.CuStaticSharedArray(Float32,200);y=CUDA.CuStaticSharedArray(Float32,200);mask=CUDA.CuStaticSharedArray(UInt8,600)
 bicluster_body!(args...,x,y,mask,Int32(i)-Int32(1));nothing
end
function launch(c)
 spec=JACC.launch_spec(threads=1,blocks=Int(c.m),shmem_size=0,sync=false)
 JACC.parallel_for(spec,Int(c.m),bicluster_kernel!,c.gene.data,c.n,c.m,c.D,c.thr,c.sample.data,c.data.data,c.score.data,c.dc.data,c.sc.data,c.tmps.data,c.tmpd.data)
 nothing
end
include("../study_common.jl")
main()
