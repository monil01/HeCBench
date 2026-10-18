using KernelAbstractions
include("../study_common.jl")
@kernel function attenuation_ka(qid,fid,flux,source,sigma,state,v,F,G)
 gid=Int32(@index(Global,Linear))-Int32(1)
 attenuate_site!(gid,qid,fid,flux,source,sigma,state,v,F,G,Int32(1))
end
const KERNEL=attenuation_ka(CUDABackend(),128)
launch(c)=KERNEL(c.qid.data,c.fid.data,c.flux.data,c.src.data,c.sig.data,c.state.data,c.scratch.data,c.F,c.G;ndrange=128*128)
simplemoc_study(launch)
