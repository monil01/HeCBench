using JACC
include("../study_common.jl")
function attenuation_jacc(i,qid,fid,flux,source,sigma,state,v,F,G)
 attenuate_site!(Int32(i)-Int32(1),qid,fid,flux,source,sigma,state,v,F,G,Int32(1))
 nothing
end
const SPEC=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=128,blocks=128,shmem_size=0,sync=false)
launch(c)=JACC.parallel_for(attenuation_jacc,SPEC,128*128,c.qid.data,c.fid.data,c.flux.data,c.src.data,c.sig.data,c.state.data,c.scratch.data,c.F,c.G)
simplemoc_study(launch)
