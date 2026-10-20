using CUDA, KernelAbstractions
import Atomix
const MODEL="ka"
const BACKEND=CUDA.CUDABackend()
include(joinpath(@__DIR__,"..","study_device.jl"))
@inline function snap_atomic!(a,i,v); Atomix.@atomic a[i] += v; nothing; end
@kernel function reset_ulisttot_entry(ulisttot,ulisttot_size)
ix=Int32(@index(Global,Linear))-Int32(1);iy=Int32(0)
reset_ulisttot(ulisttot,ulisttot_size,ix,iy)
end
@kernel function set_ulisttot_entry(ulisttot,idxu_block,num_atoms,twojmax,wself)
ix=Int32(@index(Global,Linear))-Int32(1);iy=Int32(0)
set_ulisttot(ulisttot,idxu_block,num_atoms,twojmax,wself,ix,iy)
end
@kernel function update_ulisttot_entry(rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq)
index=@index(Global,NTuple);ix=Int32(index[1])-Int32(1);iy=Int32(index[2])-Int32(1)
update_ulisttot(rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq,ix,iy)
end
@kernel function reset_ylist_entry(ylist,ylist_size)
ix=Int32(@index(Global,Linear))-Int32(1);iy=Int32(0)
reset_ylist(ylist,ylist_size,ix,iy)
end
@kernel function compute_yi_entry(idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim)
index=@index(Global,NTuple);ix=Int32(index[1])-Int32(1);iy=Int32(index[2])-Int32(1)
compute_yi(idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim,ix,iy)
end
@kernel function compute_duidrj_entry(wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag)
index=@index(Global,NTuple);ix=Int32(index[1])-Int32(1);iy=Int32(index[2])-Int32(1)
compute_duidrj(wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag,ix,iy)
end
@kernel function compute_deidrj_entry(idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max)
index=@index(Global,NTuple);ix=Int32(index[1])-Int32(1);iy=Int32(index[2])-Int32(1)
compute_deidrj(idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max,ix,iy)
end
function launch_pipeline!(c)
s=c.s;na=Int32(s.na);nb=Int32(s.nb);J=Int32(s.J);nd=Int32(s.nd);pdim=Int32(s.pdim);nz=Int32(s.nz);jdim=Int32(s.jdim);pairgrid=(cld(na,16),cld(nb,16))
reset_ulisttot_entry(BACKEND,256)(c.ut.d,na*Int32(s.nu);ndrange=na*Int32(s.nu))
set_ulisttot_entry(BACKEND,256)(c.ut.d,c.iu.d,na,J,s.s[2];ndrange=na)
update_ulisttot_entry(BACKEND,(16,16))(c.r.d,c.rc.d,c.w.d,c.parity.d,c.iu.d,c.root.d,c.u.d,c.ut.d,na,nb,Int32(1),J,pdim;ndrange=(na,nb))
CUDA.device_synchronize()
reset_ylist_entry(BACKEND,256)(c.y.d,na*nd;ndrange=na*nd)
compute_yi_entry(BACKEND,(16,16))(c.iz.d,c.zb.d,c.cg.d,c.icg.d,c.iu.d,c.id.d,c.ut.d,c.y.d,na,nz,jdim;ndrange=(na,nz))
CUDA.device_synchronize()
compute_duidrj_entry(BACKEND,(16,16))(c.w.d,c.r.d,c.rc.d,c.root.d,c.u.d,c.du.d,na,nb,J,nd,pdim,Int32(1);ndrange=(na,nb))
CUDA.device_synchronize()
compute_deidrj_entry(BACKEND,(16,16))(c.id.d,c.du.d,c.y.d,c.d.d,na,nb,J,nd;ndrange=(na,nb))
CUDA.device_synchronize()
nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study()
