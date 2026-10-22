using CUDA,KernelAbstractions
const MODEL="ka"
@kernel function set_input!(d,nr::UInt64,nc::UInt64,ld::UInt64,base::UInt64)
 i=UInt64(@index(Global,Linear))-UInt64(1)
 if i<nr*nc
  r=i%nr;c=i÷nr;s=UInt64(1)+(r+base)%UInt64(499)
  @inbounds d[Int(r+ld*c+1)]=c%s==0 ? 1f0 : 0f0
 end
end
function launch!(a,base)
 set_input!(KernelAbstractions.get_backend(a.d),256)(a.d,UInt64(a.rows),UInt64(a.cols),UInt64(a.ld),base;ndrange=a.rows*a.cols)
end
include("/tmp/hecbench-roofline-v100-20261020/batch024-completion-nz-priority/sparkler/ka/scratch/profile_source/study_common.jl")
study()
