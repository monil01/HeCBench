"""Source-output emulator + independent predecessor-aware seam objective check. No GPU work."""
from pathlib import Path
import numpy as np,json,hashlib,struct
P=Path(__file__).resolve().parent;W=H=128;K=100;I=np.iinfo(np.int32).max
rgb=np.fromfile(P/'study_source_rgb_u8.bin',np.uint8).reshape(H,W,3)
raw=(P.parent/'seam-carving-cuda/Broadway_tower.jpg').read_bytes();off=struct.unpack_from('<I',raw,10)[0];bmp=np.frombuffer(raw[off:off+W*H*3],np.uint8).reshape(H,W,3)[::-1,:,::-1];assert np.array_equal(rgb,bmp)
pixels=rgb[:,:,0].astype(np.uint32)|(rgb[:,:,1].astype(np.uint32)<<8)|(rgb[:,:,2].astype(np.uint32)<<16);pixels.tofile(P/'study_pixels_u32.bin')
def costs(image,cw):
 a=np.stack([image[:,:cw]&255,(image[:,:cw]>>8)&255,(image[:,:cw]>>16)&255],axis=2).astype(np.int32);pad=np.pad(a,((1,0),(1,1),(0,0)));left=pad[1:,:-2];right=pad[1:,2:];up=pad[:-1,1:-1];p=abs(right-left);u=p.sum(2);l=(p+abs(up-left)).sum(2);r=(p+abs(up-right)).sum(2);return l.astype(np.int16),u.astype(np.int16),r.astype(np.int16)
def dp_and_ideal(c):
 l,u,r=(z.astype(np.int32)for z in c);cw=l.shape[1];a=np.empty((H,cw),np.int32);pred=np.empty((H,cw),np.int32);a[0]=np.minimum(np.minimum(l[0],u[0]),r[0]);pred[0]=np.arange(cw)
 for row in range(1,H):
  ll=np.r_[I,a[row-1,:-1]+l[row,1:]];uu=a[row-1]+u[row];rr=np.r_[a[row-1,1:]+r[row,:-1],I];z=np.stack([ll,uu,rr]);choice=np.argmin(z,axis=0);a[row]=z[choice,np.arange(cw)];pred[row]=np.arange(cw)+choice-1
 col=int(np.argmin(a[-1]));path=np.empty(H,np.int32);path[-1]=col
 for row in range(H-1,0,-1):col=int(pred[row,col]);path[row-1]=col
 return a,path,int(a[-1].min())
def objective(path,c):
 cw=c[0].shape[1]
 if np.any(path<0)or np.any(path>=cw)or np.any(abs(np.diff(path))>1):return None
 score=int(min(z[0,path[0]]for z in c))
 for row in range(1,H):score+=int(c[int(path[row-1]-path[row])+1][row,path[row]]) # prev-left=>left index0,prevsame=>up1,prevright=>right2
 return score
summaries={}
for mode in range(3):
 img=pixels.copy();M=np.zeros((H,W),np.int32);cl=np.zeros((H,W),np.int16);cu=cl.copy();cr=cl.copy();index=np.zeros((H,W),np.int32);offset=index.copy();trace=[];quality=[];update_mismatch=[];cw=W
 if mode==1:
  cl[:],cu[:],cr[:]=costs(img,cw)
 for step in range(K):
  idealcost=costs(img,cw)
  if mode==0:cl[:,:cw],cu[:,:cw],cr[:,:cw]=idealcost
  if mode in (0,1):
   for k,z in enumerate([cl,cu,cr]):
    diff=z[:,:cw].astype(np.int32)-idealcost[k].astype(np.int32)
    if np.any(diff):update_mismatch.append({'step':step,'field':k,'mismatched':int(np.count_nonzero(diff)),'max_abs':int(abs(diff).max())})
   a,idealpath,best=dp_and_ideal([cl[:,:cw],cu[:,:cw],cr[:,:cw]]);M[:,:cw]=a;ix=int(np.argmin(a[-1]));path=np.empty(H,np.int32);path[-1]=ix;flat=M.ravel()
   for row in range(H-2,-1,-1):
    mid=ix
    if mid!=0 and flat[row*W+mid-1]<flat[row*W+ix]:ix=mid-1
    if mid!=cw and flat[row*W+mid+1]<flat[row*W+ix]:ix=mid+1
    path[row]=ix
  else:
   l,u,r=idealcost
   for row in range(H-1):
    for col in range(cw):
     candidates=[]
     if col>0:candidates.append((int(r[row+1,col-1]),col-1))
     candidates.append((int(u[row+1,col]),col))
     if col<cw-1:candidates.append((int(l[row+1,col+1]),col+1))
     cost,dest=min(candidates,key=lambda z:z[0]);index[row,col]=(row+1)*W+dest;offset[row,col]=index[row,col];M[row,col]=cost
   st=1;blocks=H//2
   while blocks:
    for by in range(blocks):
     row=by*2*st
     if row+st<H-1:
      for col in range(cw):dest=int(offset[row,col]);M[row,col]+=M.ravel()[dest];offset[row,col]=offset.ravel()[dest]
    blocks//=2;st*=2
   ix=int(np.argmin(M[0,:cw]));path=np.empty(H,np.int32)
   for row in range(H):path[row]=ix-row*W;ix=int(index.ravel()[ix])
   a,idealpath,best=dp_and_ideal(idealcost)
  score=objective(path,idealcost);idealscore=objective(idealpath,idealcost);assert idealscore==best
  quality.append({'step':step,'current_width':cw,'source_path_valid':score is not None,'source_path_forward_energy':score,'optimal_forward_energy':best,'optimality_gap':None if score is None else score-best,'approximation_expected':mode==2,'source_path':path.tolist()if step==0 else None,'ideal_path':idealpath.tolist()if step==0 else None})
  trace.append(path.copy());nextimg=img.copy()
  for row in range(H):
   for col in range(cw-1):nextimg[row,col]=img[row,col+int(col>=path[row])]
  img=nextimg
  if mode==1:
   nl,nu,nr=(z.copy()for z in [cl,cu,cr]);full=costs(img,cw-1)
   for row in range(H):
    seam=int(path[row])
    for col in range(cw-1):
     if seam-2<=col<=seam+1:
      nl[row,col]=full[0][row,col];nu[row,col]=full[1][row,col];nr[row,col]=full[2][row,col]
     else:
      old=col+int(col>seam+1);nl[row,col]=cl[row,old];nu[row,col]=cu[row,old];nr[row,col]=cr[row,old]
   cl,cu,cr=nl,nu,nr
  cw-=1
 prefix=np.r_[pixels.ravel(),np.array([W,H,K,mode,1,W-K+1,W-K],np.uint32)];pieces=[img[:,:cw].ravel().astype(np.uint32),path.astype(np.uint32)]
 if mode<2:
  pieces.append(M[:,:cw+1].ravel().astype(np.uint32));costw=cw if mode==1 else cw+1;pieces.extend(z[:,:costw].ravel().astype(np.uint32)for z in [cl,cu,cr])
 else:pieces.extend(z[:H-1,:cw+1].ravel().astype(np.uint32)for z in [M,index,offset])
 out=np.concatenate([prefix]+pieces);out.astype('<u4').tofile(P/f'study_expected_mode{mode}_u32.bin');np.array(trace,dtype='<i4').tofile(P/f'study_trace_mode{mode}_i32.bin');bad=[q for q in quality if not q['source_path_valid']or q['optimality_gap']!=0];summaries[str(mode)]={'original_fixture':'128x128/100seams','mode':['standard','update','approx'][mode],'output_width':cw,'prefix_count':len(prefix),'artifact_count':len(out),'invalid_path_steps':sum(not q['source_path_valid']for q in quality),'positive_optimality_gap_steps':sum(q['optimality_gap']is not None and q['optimality_gap']>0 for q in quality),'max_optimality_gap':max(q['optimality_gap']or 0 for q in quality),'first_nonoptimal_or_invalid_step':bad[0]if bad else None,'update_vs_recomputed_cost_differences':update_mismatch,'CPUtrajectory_quality':quality,'minimum_energy_source_quality_failure':mode<2 and bool(bad),'approx_gap_is_documented_compromise':mode==2,'oracle_strength':'SourcebehaviorCPUemulator exactcomputationaloracle plusindependentlypredecessor-awareDP objectivecheck; nohumanperceptualqualityclaim'}
(P/'source_trajectory_quality.json').write_text(json.dumps({'numpy':np.__version__,'source_STB_RGB_matches_independent_BMP_decoder':True,'source_alpha_byte':'Unuseduninitializedsourceuchar4.w iscanonical0inthisRGBfixture; allkernelsreadRGBonly.','cases':summaries},indent=2)+'\n');(P/'fixture_provenance.json').write_text(json.dumps({'source_image_sha256':hashlib.sha256((P.parent/'seam-carving-cuda/Broadway_tower.jpg').read_bytes()).hexdigest(),'generator_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest()for f in ['decode_source_image.cpp','generate_study_oracles.py']},'files':{q.name:{'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size}for q in P.glob('study_*.bin')}},indent=2)+'\n');print({m:{k:v for k,v in z.items()if k not in ('CPUtrajectory_quality','first_nonoptimal_or_invalid_step','update_vs_recomputed_cost_differences')}for m,z in summaries.items()})
