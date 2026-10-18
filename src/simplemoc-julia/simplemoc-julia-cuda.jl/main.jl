include("../study_common.jl")
function attenuation_cuda(qid,fid,flux,source,sigma,state,v,F,G)
 gid=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
 attenuate_site!(gid,qid,fid,flux,source,sigma,state,v,F,G,Int32(1))
 return
end
launch(c)=@cuda threads=128 blocks=128 attenuation_cuda(c.qid.data,c.fid.data,c.flux.data,c.src.data,c.sig.data,c.state.data,c.scratch.data,c.F,c.G)
simplemoc_study(launch)
