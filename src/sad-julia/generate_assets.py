#!/usr/bin/env python3
"""Independent full valid-window RGB SAD and explicit source-prefix witness."""
from pathlib import Path
import numpy as np,struct,json,hashlib
P=Path(__file__).resolve().parent

def read_bmp(path):
 b=path.read_bytes();assert b[:2]==b'BM';offset=struct.unpack_from('<I',b,10)[0];w,h=struct.unpack_from('<ii',b,18);assert w>0 and h>0 and struct.unpack_from('<H',b,28)[0]==24 and struct.unpack_from('<I',b,30)[0]==0
 stride=(w*3+3)//4*4;assert len(b)>=offset+stride*h
 return np.frombuffer(b[offset:offset+stride*h],dtype='u1').reshape(h,stride)[:,:w*3].reshape(h,w,3)[::-1,:,::-1].copy()

def valid_sad(image,kernel):
 h,w,_=image.shape;kh,kw,_=kernel.shape;vh=h-kh+1;vw=w-kw+1;total=np.zeros((vh,vw),dtype='i4')
 for y in range(kh):
  for x in range(kw):total+=np.abs(image[y:y+vh,x:x+vw].astype('i2')-kernel[y,x].astype('i2')).sum(axis=2,dtype='i4')
 # All declared sums <2^24 and integer-quotient gap >Float32 division halfULP.
 assert int(total.max())<2**24
 out=total//(kh*kw);assert np.array_equal(out,(total.astype('f4')/np.float32(kh*kw)).astype('i4'))
 return out.astype('<i4')

def original_prefix_sad(image,kernel,n):
 h,w,_=image.shape;kh,kw,_=kernel.shape;rows=(n+w-1)//w;total=np.zeros((rows,w),dtype='i4')
 for y in range(kh):
  for x in range(kw):
   rh=min(rows,h-y);rw=w-x
   if rh>0 and rw>0:total[:rh,:rw]+=np.abs(image[y:y+rh,x:x+rw].astype('i2')-kernel[y,x].astype('i2')).sum(axis=2,dtype='i4')
 return (total.ravel()[:n].astype('f4')/np.float32(kh*kw)).astype('<i4')

def main():
 cases=[(read_bmp(P/'original_assets/coins.bmp'),read_bmp(P/'original_assets/coin.bmp'),'original_DVC_real_images',[])]
 rng=np.random.default_rng(731)
 for h,w,kh,kw,positions,name in [(32,32,17,17,[(15,15)],'known_bottomright_match'),(32,64,17,17,[(0,0),(15,47)],'two_exact_matches')]:
  image=rng.integers(0,256,size=(h,w,3),dtype='u1');kernel=rng.integers(0,256,size=(kh,kw,3),dtype='u1')
  for y,x in positions:image[y:y+kh,x:x+kw]=kernel
  cases.append((image,kernel,name,positions))
 cases.extend([(np.full((40,40,3),47,dtype='u1'),np.full((9,9,3),47,dtype='u1'),'constant_all_matches',[]),(np.full((17,17,3),255,dtype='u1'),np.zeros((1,1,3),dtype='u1'),'threshold_no_hit_partial_group',[])])
 plan=[]
 for k,(image,kernel,name,positions) in enumerate(cases):
  h,w,_=image.shape;kh,kw,_=kernel.shape;sad=valid_sad(image,kernel);n=sad.size;minimum=min(20,int(sad.min()));count=int((sad==minimum).sum());source=original_prefix_sad(image,kernel,n);source_min=min(20,int(source.min()));source_count=int((source==source_min).sum())
  for label,a in [('image',image),('kernel',kernel),('oracle',sad),('original_oracle',source)]:a.tofile(P/f'{label}{k}.bin')
  d={'width':w,'height':h,'kernel_width':kw,'kernel_height':kh,'sad_count':n,'valid_width':w-kw+1,'valid_height':h-kh+1,'kind':name,'intended_min_capped_at20':minimum,'intended_occurrences_at_capped_min':count,'uncapped_minimum':int(sad.min()),'known_inserted_matches':positions,'original_prefix_field_mismatches':int(np.sum(source!=sad.ravel())),'predicted_original_min_capped20':source_min,'predicted_original_occurrences':source_count}
  plan.append(d);print(json.dumps(d),flush=True)
 d={'cases':plan,'atol':0,'rtol':0,'source_timing':'Preserve source3kernel launch+wholecontextcompletion interval; reinitialize min20/count0 outsideROI eachinternalrepeat; sumR3repeatregionmeans across5cases; transfers/allocation/fullcapture/independentoracle outsideROI.','scientific_contract':'Full valid-window normalized RGB L1 template distance, floor normalization, threshold-capped20minimum and exactoccurrencecount, including real original DVCimages, bottomright/two/constant/nohit controls. This is templateintensitydistance notgeneralvisiondetectionaccuracy.','source_coordinate_witness':'Originalkernel indexes row*image_width+col into compactvalidwindowextent and includes partialrightimagewindows; independentlypredictedcompletewrongfield and actual nativecapture required beforecorrection.','normalization':'Int32 sum<2^24; exactinteger quotient matches sourceFloat32division/truncation at everyactualdeclared location.','rng':'numpy.default_rng seed731, exact generated hashes below','files':{f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for label in ['image','kernel','oracle','original_oracle'] for f in sorted(P.glob(label+'[0-9].bin'))}}
 (P/'study_plan.json').write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
