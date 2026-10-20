using CUDA
import JACC, Atomix
JACC.@init_backend
const MODEL="jacc"
include(joinpath(@__DIR__,"..","study_device.jl"))
@inline function snap_atomic!(a,i,v); Atomix.@atomic a[i] += v; nothing; end
function reset_ulisttot_entry(i,ulisttot,ulisttot_size)
ix=Int32(i)-Int32(1);iy=Int32(0)
reset_ulisttot(ulisttot,ulisttot_size,ix,iy)
return nothing
end
function set_ulisttot_entry(i,ulisttot,idxu_block,num_atoms,twojmax,wself)
ix=Int32(i)-Int32(1);iy=Int32(0)
set_ulisttot(ulisttot,idxu_block,num_atoms,twojmax,wself,ix,iy)
return nothing
end
function update_ulisttot_entry(i,j,rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq)
ix=Int32(i)-Int32(1);iy=Int32(j)-Int32(1)
update_ulisttot(rij,rcutij,wj,ulist_parity,idxu_block,rootpqarray,ulist,ulisttot,num_atoms,num_nbor,switch_flag,twojmax,jdimpq,ix,iy)
return nothing
end
function reset_ylist_entry(i,ylist,ylist_size)
ix=Int32(i)-Int32(1);iy=Int32(0)
reset_ylist(ylist,ylist_size,ix,iy)
return nothing
end
function compute_yi_entry(i,j,idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim)
ix=Int32(i)-Int32(1);iy=Int32(j)-Int32(1)
compute_yi(idxz,idxzbeta,cglist,idxcg_block,idxu_block,idxdu_block,ulisttot,ylist,num_atoms,idxz_max,jdim,ix,iy)
return nothing
end
function compute_duidrj_entry(i,j,wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag)
ix=Int32(i)-Int32(1);iy=Int32(j)-Int32(1)
compute_duidrj(wj,rij,rcutij,rootpqarray,ulist,dulist,num_atoms,num_nbor,twojmax,idxdu_max,jdimpq,switch_flag,ix,iy)
return nothing
end
function compute_deidrj_entry(i,j,idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max)
ix=Int32(i)-Int32(1);iy=Int32(j)-Int32(1)
compute_deidrj(idxdu_block,dulist,ylist,dedr,num_atoms,num_nbor,twojmax,idxdu_max,ix,iy)
return nothing
end
function launch_pipeline!(c)
s=c.s;na=Int32(s.na);nb=Int32(s.nb);J=Int32(s.J);nd=Int32(s.nd);pdim=Int32(s.pdim);nz=Int32(s.nz);jdim=Int32(s.jdim);pairgrid=(cld(na,16),cld(nb,16))
spec=JACC.launch_spec(threads=256,blocks=cld(na*Int32(s.nu),256),shmem_size=0,sync=false)
JACC.parallel_for(spec,cld(na*Int32(s.nu),256)*256,reset_ulisttot_entry,c.ut.d,na*Int32(s.nu))
spec=JACC.launch_spec(threads=256,blocks=cld(na,256),shmem_size=0,sync=false)
JACC.parallel_for(spec,cld(na,256)*256,set_ulisttot_entry,c.ut.d,c.iu.d,na,J,s.s[2])
spec=JACC.launch_spec(threads=(16,16),blocks=pairgrid,shmem_size=0,sync=false)
JACC.parallel_for(spec,(pairgrid[1]*16,pairgrid[2]*16),update_ulisttot_entry,c.r.d,c.rc.d,c.w.d,c.parity.d,c.iu.d,c.root.d,c.u.d,c.ut.d,na,nb,Int32(1),J,pdim)
CUDA.device_synchronize()
spec=JACC.launch_spec(threads=256,blocks=cld(na*nd,256),shmem_size=0,sync=false)
JACC.parallel_for(spec,cld(na*nd,256)*256,reset_ylist_entry,c.y.d,na*nd)
spec=JACC.launch_spec(threads=(16,16),blocks=(cld(na,16),cld(nz,16)),shmem_size=0,sync=false)
JACC.parallel_for(spec,(cld(na,16)*16,cld(nz,16)*16),compute_yi_entry,c.iz.d,c.zb.d,c.cg.d,c.icg.d,c.iu.d,c.id.d,c.ut.d,c.y.d,na,nz,jdim)
CUDA.device_synchronize()
spec=JACC.launch_spec(threads=(16,16),blocks=pairgrid,shmem_size=0,sync=false)
JACC.parallel_for(spec,(pairgrid[1]*16,pairgrid[2]*16),compute_duidrj_entry,c.w.d,c.r.d,c.rc.d,c.root.d,c.u.d,c.du.d,na,nb,J,nd,pdim,Int32(1))
CUDA.device_synchronize()
spec=JACC.launch_spec(threads=(16,16),blocks=pairgrid,shmem_size=0,sync=false)
JACC.parallel_for(spec,(pairgrid[1]*16,pairgrid[2]*16),compute_deidrj_entry,c.id.d,c.du.d,c.y.d,c.d.d,na,nb,J,nd)
CUDA.device_synchronize()
nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study()
