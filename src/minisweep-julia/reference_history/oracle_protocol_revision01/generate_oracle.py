
from pathlib import Path
import numpy as np
import json,hashlib,sys,math
P=Path(__file__).resolve().parent
def sf(x,y,z):return 1<<((x+3*y+7*z+2)&3)
def ef(e):return 1<<(((e*1366+150889)%714025)&3)
def uf(u):return 1<<(((u*741+66037)%312500)&3)
def matrices(na):
 A=np.zeros((8,4,na),np.float64);M=np.zeros((8,na,4),np.float64)
 for ia in range(na):
  q,t=divmod(ia+1,4);A[:,3,ia]+=q
  if t:A[:,0,ia]-=1;A[:,t,ia]+=1
 for im in range(2):
  for ia in range(na):
   v=21+(im+4*ia)%17;A[:,im,ia]-=v;A[:,im+1,ia]+=2*v;A[:,im+2,ia]-=v
 for im in range(4):
  q,t=divmod(im+1,na);M[:,na-1,im]+=q
  if t:M[:,0,im]-=1;M[:,t,im]+=1
 for im in range(4):
  for ia in range(na-2):
   v=37+(im+4*ia)%19;M[:,ia,im]-=v;M[:,ia+1,im]+=2*v;M[:,ia+2,im]-=v
 M/=8
 for ia in range(na):M[:,ia,:]/=1<<(ia&7)
 return A,M
def generate(dims,perturbed):
 nx,ny,nz,ne,na=dims;A,M=matrices(na)
 vi=np.empty((nz,ne,ny,nx,4,4),np.float64)
 for z in range(nz):
  for e in range(ne):
   for y in range(ny):
    for x in range(nx):
     for u in range(4):
      for im in range(4):
       vi[z,e,y,x,u,im]=(im+1)*sf(x,y,z)*ef(e)*uf(u)
       if perturbed:vi[z,e,y,x,u,im]+=((im*im+3*u*x+5*y+7*z+11*e)%17-8)/64
 # Independent CPU topological octant traversal + BLAS small matrix products, not GPUwavefront transcription.
 vo=np.zeros_like(vi);xy=np.empty((8,ny,nx,ne,4,na));xz=np.empty((8,nz,nx,ne,4,na));yz=np.empty((8,nz,ny,ne,4,na));local=np.empty_like(xy)
 scale=np.array([[[ef(e)*uf(u)*(a+1)*(1<<(a&7))for a in range(na)]for u in range(4)]for e in range(ne)],np.float64)
 wx=.25-1/np.array([1<<(a&7)for a in range(na)])
 for o in range(8):
  dx=-1 if o&1 else 1;dy=-1 if o&2 else 1;dz=-1 if o&4 else 1
  zb=-1 if dz==1 else nz;yb=-1 if dy==1 else ny;xb=-1 if dx==1 else nx
  for y in range(ny):
   for x in range(nx):xy[o,y,x]=scale*sf(x,y,zb)*(o+1)
  for z in range(nz):
   for x in range(nx):xz[o,z,x]=scale*sf(x,yb,z)*(o+1)
  for z in range(nz):
   for y in range(ny):yz[o,z,y]=scale*sf(xb,y,z)*(o+1)
  for z in (range(nz)if dz==1 else range(nz-1,-1,-1)):
   for y in (range(ny)if dy==1 else range(ny-1,-1,-1)):
    for x in (range(nx)if dx==1 else range(nx-1,-1,-1)):
     angle=vi[z,:,y,x]@A[o]
     incoming=xy[o,y,x]*.5/sf(x,y,z-dz)+xz[o,z,x]*.25/sf(x,y-dy,z)+yz[o,z,y]*wx/sf(x-dx,y,z)
     result=(angle/sf(x,y,z)+incoming/(o+1))*sf(x,y,z)
     local[o,y,x]=result
     xy[o,y,x]=result*(o+1);xz[o,z,x]=result*(o+1);yz[o,z,y]=result*(o+1)
     vo[z,:,y,x]+=result@M[o]
 return A,M,vi,vo,xy,xz,yz,local
def main():
 cases=[];files={}
 for case,dims in enumerate([(5,5,5,30,33),(2,3,2,2,5)],1):
  A,M,vi,*outputs=generate(dims,case==2)
  if case==1:assert np.array_equal(vi,outputs[0])
  else:assert np.max(np.abs(vi-outputs[0]))>1
  inputv=np.concatenate([A.ravel(),M.ravel(),vi.ravel()]);out=np.concatenate([v.ravel()for v in outputs]);assert np.all(np.isfinite(out))
  for name,v in [(f'study_case{case}_f64.bin',inputv),(f'oracle_case{case}_f64.bin',out)]:
   b=v.astype('<f8').tobytes();(P/name).write_bytes(b);files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  cases.append(dict(case=case,dims=list(dims),input_count=len(inputv),output_count=len(out),state_count=vi.size,matrix_count=A.size,face_counts=[a.size for a in outputs[1:]],manufactured_identity_exact=case==1,max_output_state_minus_input=float(np.max(np.abs(outputs[0]-vi))),input_normsq=float(np.sum(vi*vi)),method='IndependentCPUtopologicallexicographicupwindtraversal, NumPyBLASsmallmatrixproducts; sourceaffineidentity separatelychecked,nonaffineperturbationdistinguishesidentitystub.'))
 (P/'fixture_provenance.json').write_text(json.dumps(dict(cases=cases,files=files,oracle='Fullstate/threefacefields/persistentanglescratch aftercomplete8octants; CPUtopologicalorder differsGPUwavefrontorder; source-matrixconstruction preserved; no physicalconservationclaim.',NumPy=np.__version__,Python=sys.version,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_unknown_scale_note='Host66037anddevice60037offsets differ6000butmod4pow2scalesidentical forNU4; no silentlycorrecteddeviceformula.',timed_input_reuse='Source never swapsdevicevi/vo; repeatsrecomputesameinput, nottimeevolution'),indent=2)+'\n')
 print(json.dumps(cases))
if __name__=='__main__':main()
