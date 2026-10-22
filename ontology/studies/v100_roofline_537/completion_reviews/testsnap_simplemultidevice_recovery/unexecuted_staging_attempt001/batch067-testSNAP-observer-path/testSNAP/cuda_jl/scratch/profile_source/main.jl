using CUDA
const MODEL="cuda_jl"
include("/home/imo/HeCBench_julia/src/testSNAP-julia/study_device.jl")
@inline function snap_atomic!(a,i,v); CUDA.@atomic a[i] += v; nothing; end
function reset_ulisttot_entry(ulisttot,ulisttot_size)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
reset_ulisttot(ulisttot,ulisttot_size,ix,iy)
return nothing
end
function set_ulisttot_entry(ulisttot,idxu_block,num_atoms,twojmax,wself)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
set_ulisttot(ulisttot,idxu_block,num_atoms,twojmax,wself,ix,iy)
return nothing
end
function update_ulisttot_entry(rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
update_ulisttot(rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq,ix,iy)
return nothing
end
function reset_ylist_entry(ylist,ylist_size)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
reset_ylist(ylist,ylist_size,ix,iy)
return nothing
end
function compute_yi_entry(idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
compute_yi(idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim,ix,iy)
return nothing
end
function compute_duidrj_entry(wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
compute_duidrj(wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag,ix,iy)
return nothing
end
function compute_deidrj_entry(idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max)
ix=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1)
iy=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1)
compute_deidrj(idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max,ix,iy)
return nothing
end
function launch_pipeline!(c)
s=c.s;na=Int32(s.na);nb=Int32(s.nb);J=Int32(s.J);nd=Int32(s.nd);pdim=Int32(s.pdim);nz=Int32(s.nz);jdim=Int32(s.jdim);pairgrid=(cld(na,16),cld(nb,16))
CUDA.@cuda threads=256 blocks=cld(na*Int32(s.nu),256) reset_ulisttot_entry(c.ut.d,na*Int32(s.nu))
CUDA.@cuda threads=256 blocks=cld(na,256) set_ulisttot_entry(c.ut.d,c.iu.d,na,J,s.s[2])
CUDA.@cuda threads=(16,16) blocks=pairgrid update_ulisttot_entry(c.r.d,c.rc.d,c.w.d,c.parity.d,c.iu.d,c.root.d,c.u.d,c.ut.d,na,nb,Int32(1),J,pdim)
CUDA.device_synchronize()
CUDA.@cuda threads=256 blocks=cld(na*nd,256) reset_ylist_entry(c.y.d,na*nd)
CUDA.@cuda threads=(16,16) blocks=(cld(na,16),cld(nz,16)) compute_yi_entry(c.iz.d,c.zb.d,c.cg.d,c.icg.d,c.iu.d,c.id.d,c.ut.d,c.y.d,na,nz,jdim)
CUDA.device_synchronize()
CUDA.@cuda threads=(16,16) blocks=pairgrid compute_duidrj_entry(c.w.d,c.r.d,c.rc.d,c.root.d,c.u.d,c.du.d,na,nb,J,nd,pdim,Int32(1))
CUDA.device_synchronize()
CUDA.@cuda threads=(16,16) blocks=pairgrid compute_deidrj_entry(c.id.d,c.du.d,c.y.d,c.d.d,na,nb,J,nd)
CUDA.device_synchronize()
nothing
end
include("/tmp/hecbench-roofline-v100-20261020/batch067-testSNAP-observer-path/testSNAP/cuda_jl/scratch/profile_source/study_common.jl")
study()
