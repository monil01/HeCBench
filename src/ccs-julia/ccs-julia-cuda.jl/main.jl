using CUDA
include("../study_device.jl")
function bicluster_kernel!(args...)
 x=CUDA.CuStaticSharedArray(Float32,200);y=CUDA.CuStaticSharedArray(Float32,200);mask=CUDA.CuStaticSharedArray(UInt8,600)
 bicluster_body!(args...,x,y,mask,Int32(CUDA.blockIdx().x)-Int32(1));return
end
launch(c)=CUDA.@cuda threads=1 blocks=c.m bicluster_kernel!(c.gene.data,c.n,c.m,c.D,c.thr,c.sample.data,c.data.data,c.score.data,c.dc.data,c.sc.data,c.tmps.data,c.tmpd.data)
include("../study_common.jl")
main()
