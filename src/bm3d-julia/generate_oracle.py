from pathlib import Path
import json,hashlib,math,sys
import numpy as np
from PIL import Image,__version__ as pillow_version
P=Path(__file__).resolve().parent;R=P.parent.parent
W,H,BX,BY,K,N=112,40,64,12,8,16;G=BX*BY;SX,SY=W-7,H-7
src=R/'src/bm3d-cuda/noised-bufferfly-20.png';clean=R/'src/bm3d-cuda/bufferfly.png'
image=np.array(Image.open(src),dtype=np.uint8)[:H,:W,0].copy();reference=np.array(Image.open(clean),dtype=np.uint8)[:H,:W,0].copy();image.tofile(P/'study_input_u8.bin');reference.tofile(P/'study_clean_u8.bin')
quarter=np.array([[.1924,.2989,.3846,.4325],[.2989,.4642,.5974,.6717],[.3846,.5974,.7688,.8644],[.4325,.6717,.8644,.9718]],dtype=np.float32);win=np.empty((8,8),np.float32)
for y in range(8):
 for x in range(8):win[y,x]=quarter[min(y,7-y),min(x,7-x)]
win.tofile(P/'study_window_f32.bin')
Q=np.array([[(1/math.sqrt(8) if u==0 else .5)*math.cos(math.pi*(2*x+1)*u/16) for x in range(8)]for u in range(8)])
patches=np.lib.stride_tricks.sliding_window_view(image.astype(np.int64),(8,8));stacks=np.zeros((G,15),np.uint16);counts=np.zeros(G,np.uint32);weights=np.zeros(G,np.float32);nzcounts=np.zeros(G,np.uint32);post=np.zeros((G,N,8,8),np.float32);num=np.zeros((H,W));den=np.zeros((H,W));minmargin=math.inf;valid=0
for gy in range(BY):
 for gx in range(BX):
  px=gx*3;py=gy*3
  if SX<=px<SX+2:px=SX-1
  if SY<=py<SY+2:py=SY-1
  if px>=SX or py>=SY:continue
  g=gy*BX+gx;valid+=1;base=(gx//32)*96;left=max(-19,-base);ys=range(max(-19,-py),min(19,SY-1-py)+1);xs=range(left,min(19,SX-1-px)+1);records=[]
  for dy in ys:
   if not xs:continue
   xx=np.array([px+dx for dx in xs]);ssd=np.sum((patches[py+dy,xx]-patches[py,px])**2,axis=(1,2))
   for dx,d in zip(xs,ssd):
    if dx==0 and dy==0:continue
    if d<160000:records.append(((int(d)>>2)<<16 | (dy&255)<<8 | (dx&255),dx,dy))
  records.sort();chosen=records[:15]
  for z,(key,dx,dy)in enumerate(chosen):stacks[g,z]=key&65535
  depth=2**int(math.floor(math.log2(len(chosen)+1)));counts[g]=depth-1;coords=[(px,py)]+[(px+dx,py+dy)for _,dx,dy in chosen[:depth-1]]
  group=np.stack([image[y:y+8,x:x+8].astype(float)for x,y in coords]);freq=np.stack([Q@a@Q.T for a in group]);had=np.array([[(-1)**((a&b).bit_count())for b in range(depth)]for a in range(depth)],dtype=float);f=np.einsum('ab,byx->ayx',had,freq);threshold=float(np.float32(2.7))*math.sqrt(depth*625);minmargin=min(minmargin,float(np.min(np.abs(np.abs(f)-threshold))));mask=np.abs(f)>=threshold;nz=int(np.count_nonzero(mask));nzcounts[g]=max(nz,1);weights[g]=np.float32(1/max(nz,1));filtered=np.einsum('ab,byx->ayx',had,np.where(mask,f,0))/depth;recon=np.stack([Q.T@a@Q for a in filtered]);post[g,:depth]=recon.astype(np.float32)
  for a,(x,y)in zip(recon,coords):num[y:y+8,x:x+8]+=a*win*float(weights[g]);den[y:y+8,x:x+8]+=win*float(weights[g])
assert np.all(den>0)
out=np.clip(np.rint(num/den),0,255).astype(np.uint8)
for name,a in [('stacks_u16',stacks),('counts_u32',counts),('nonzero_u32',nzcounts),('weights_f32',weights),('post_f32',post),('num_f32',num.astype(np.float32)),('den_f32',den.astype(np.float32)),('output_u8',out)]:a.tofile(P/('oracle_'+name+'.bin'))
files=['study_input_u8.bin','study_clean_u8.bin','study_window_f32.bin']+[p.name for p in P.glob('oracle_*.bin')]
meta={'generator':'generate_oracle.py','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy':np.__version__,'pillow':pillow_version,'source_images':[{'path':str(z.relative_to(R)),'sha256':hashlib.sha256(z.read_bytes()).hexdigest()}for z in [src,clean]],'crop':'x0..111/y0..39/redplane','valid_reference_patches':valid,'minimum_threshold_distance_Float64':minmargin,'guard_policy':'exactFloat32sentinel1515.25,16prefix/suffixperallocation','CPU_quality_MSE_noisy':float(np.mean((image.astype(float)-reference)**2)),'CPU_quality_MSE_denoised':float(np.mean((out.astype(float)-reference)**2)),'files':{n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in files}};(P/'fixture_provenance.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
