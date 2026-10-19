using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel function bicluster_kernel!(gene,n,m,D,thr,sample,data,score,dc,sc,tmps,tmpd)
 x=@localmem Float32 (200,)
 y=@localmem Float32 (200,)
 mask=@localmem UInt8 (600,)
 k=@index(Group,Linear)
 bicluster_body!(gene,n,m,D,thr,sample,data,score,dc,sc,tmps,tmpd,x,y,mask,Int32(k)-Int32(1))
end
launch(c)=bicluster_kernel!(CUDABackend(),1)(c.gene.data,c.n,c.m,c.D,c.thr,c.sample.data,c.data.data,c.score.data,c.dc.data,c.sc.data,c.tmps.data,c.tmpd.data;ndrange=Int(c.m))
include("../study_common.jl")
main()
