"""Independent Float64 array geometry and visibility oracle, CPU only.
No source C++/Julia geometry routines or GPU outputs are imported.
"""
from pathlib import Path
import numpy as np,json,hashlib,math
P=Path(__file__).resolve().parent;W=H=256;scene=np.array([-2,0,-3.5,.5,-.5,0,-3,.5,1,0,-2.2,.5,0,-.5,0,0,1,0],dtype='<f4');scene.tofile(P/'study_scene_f32.bin');spheres=scene[:12].astype(np.float64).reshape(3,4);pp=scene[12:15].astype(np.float64);pn=scene[15:18].astype(np.float64);eps=float(np.float32(.0001));N=W*H
norm=lambda a:a/np.linalg.norm(a,axis=1)[:,None]
def intersect(org,d):
 count=len(d);best=np.full(count,np.inf);hp=np.zeros((count,3));hn=np.zeros((count,3));hit=np.zeros(count,bool)
 for q in spheres:
  off=org-q[:3];B=np.einsum('ij,ij->i',off,d);C=np.einsum('ij,ij->i',off,off)-q[3]**2;disc=B*B-C;t=-B-np.sqrt(np.maximum(disc,0));take=(disc>0)&(t>0)&(t<best);best[take]=t[take];hit[take]=True;hp[take]=org[take]+d[take]*t[take,None];hn[take]=norm(hp[take]-q[:3])
 denom=d@pn;num=-(org@pn-pp@pn);t=np.divide(num,denom,out=np.full(count,np.inf),where=np.abs(denom)>=1e-17);take=(np.abs(denom)>=1e-17)&(t>0)&(t<best);best[take]=t[take];hit[take]=True;hp[take]=org[take]+d[take]*t[take,None];hn[take]=pn
 return hit,hp,hn,best
def next_rng(x):
 x=x^(x>>np.uint32(6));x=x^(x<<np.uint32(17));x=x^(x>>np.uint32(9));bits=(x&np.uint32((1<<23)-1))|np.uint32(0x3f800000);return x,(bits.view(np.float32)-np.float32(1)).astype(np.float64)
x=np.tile(np.arange(W),H);y=np.repeat(np.arange(H),W);states=np.arange(N,dtype=np.uint32);sums=np.zeros(N);counts=[];maxnorm=0.;positive=True
for v in range(2):
 for u in range(2):
  dirs=norm(np.stack([(x+u/2-W/2)/(W/2),-(y+v/2-H/2)/(H/2),-np.ones(N)],axis=1));hits,points,normals,t=intersect(np.zeros((N,3)),dirs);active=np.flatnonzero(hits);n=normals[active];p=points[active]+eps*n;maxnorm=max(maxnorm,float(np.max(abs(np.linalg.norm(n,axis=1)-1))));positive&=bool(np.all(t[active]>0));anchor=np.zeros_like(n);axis=np.where(abs(n[:,0])<.6,0,np.where(abs(n[:,1])<.6,1,np.where(abs(n[:,2])<.6,2,0)));anchor[np.arange(len(n)),axis]=1;b0=norm(np.cross(anchor,n));b1=norm(np.cross(n,b0));occluded=np.zeros(len(n),np.int32);state=states[active].copy()
  for sample in range(64):
   state,r0=next_rng(state);state,r1=next_rng(state);theta=np.sqrt(r0);phi=2*math.pi*r1;lx=np.cos(phi)*theta;ly=np.sin(phi)*theta;lz=np.sqrt(1-theta*theta);direction=lx[:,None]*b0+ly[:,None]*b1+lz[:,None]*n;occ,_,_,_=intersect(p,direction);occluded+=occ
  states[active]=state;visibility=64-occluded;sums[active]+=visibility/64;counts.append({'subpixel':[u,v],'primary_hits':int(hits.sum()),'visibility_votes':int(visibility.sum()),'total_visibility_trials':int(len(active)*64)});print(counts[-1],flush=True)
img=np.clip(np.floor(sums/4*255.5),0,255).astype(np.uint8);rgb=np.repeat(img,3);assert np.all(rgb[:W*3]==0);assert np.count_nonzero(img)>1000 and np.count_nonzero(img==0)>1000;rgb.tofile(P/'study_expected_u8.bin');report={'numpy':np.__version__,'mathematical_method':'IndependentFloat64analyticquadratic/planeintersection, geometricnormals/orthonormalbasis, sourceUInt32xorshiftuniformbits, truepi cosineweightedhemispherevisibility andquantization. DoesnotreuseGPU/C++/Juliasourcegeometry.','source_precision_boundary':'Scenepositions/radii/epsilon sourcedFloat32bits thenwidened; primarygridexactrational; stochasticdirections useFloat64sqrt/trig/truepi, intendedhighprecisionoracle, notsourcefloattranscription.','contract':'AllRGBcodeswithinpredeclared1graylevel acrossCPU/GPU; exactchannels/sky/guards/scene separately; no statisticalCI/perceptualquality claim.','subpixels':counts,'scientific_invariants':{'positive_primary_intersection_distances':positive,'maximum_surface_normal_norm_error':maxnorm,'visibility_vote_count_in_0_64':True,'image_nontrivial':True,'sky_top_row_exact_black':True,'RGB_channels_exactly_equal':True,'source_rng_seed0_absorbing_zero_stream':'Preserved; do notclaimindependent/unbiasedquality fromthisPRNG.'},'output_gray_histogram':np.bincount(img,minlength=256).tolist(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':{q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in [P/'study_scene_f32.bin',P/'study_expected_u8.bin']}}
(P/'fixture_provenance.json').write_text(json.dumps(report,indent=2)+'\n');print('CPUgeometryoracleCOMPLETE',len(rgb),flush=True)
