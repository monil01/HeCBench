#!/usr/bin/env python3
"""Independent exact pixel-patch Haar oracle and source-policy grouping."""
from pathlib import Path
import struct,json,zlib,math,hashlib
P=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def rnd(x):return int(x+(.5 if x>=0 else-.5))
def group(rs):
 n=len(rs);par=list(range(n))
 def root(i):
  while par[i]!=i:i=par[i]
  return i
 for i,a in enumerate(rs):
  for j,b in enumerate(rs):
   d=f32(f32(f32(.4)*(min(a[2],b[2])+min(a[3],b[3])))*.5)
   if all(abs(x-y)<=d for x,y in zip((a[0],a[1],a[0]+a[2],a[1]+a[3]),(b[0],b[1],b[0]+b[2],b[1]+b[3]))):par[root(j)]=root(i)
 classes={}
 for i,a in enumerate(rs):classes.setdefault(root(i),[]).append(a)
 av=[]
 for v in classes.values():
  inv=f32(1/len(v));av.append((tuple(rnd(f32(sum(a[k]for a in v)*inv))for k in range(4)),len(v)))
 out=[]
 for i,(a,na)in enumerate(av):
  if na<=1:continue
  keep=True
  for j,(b,nb)in enumerate(av):
   if i==j or nb<=1:continue
   dx=rnd(f32(b[2]*f32(.4)));dy=rnd(f32(b[3]*f32(.4)))
   if a[0]>=b[0]-dx and a[1]>=b[1]-dy and a[0]+a[2]<=b[0]+b[2]+dx and a[1]+a[3]<=b[1]+b[3]+dy and(nb>max(3,na)or na<3):keep=False;break
  if keep:out.append(a)
 return out

def oracle(w,h,img,mode,rects,stages,weights,trees,a1,a2,sth):
 trace=[];candidates=[];factor=f32(1);scales=[]
 def vec(v):trace.append(len(v));trace.extend(v)
 def flat(rs):return [x for a in rs for x in a]
 while True:
  sw=int(f32(w/factor));sh=int(f32(h/factor));ww=rnd(f32(24*factor))
  if sw<24 or sh<24:break
  xr=(w<<16)//sw+1;yr=(h<<16)//sh+1
  pix=[img[((y*yr)>>16)*w+((x*xr)>>16)]for y in range(sh)for x in range(sw)]
  sums=[];sqs=[]
  for y in range(sh):
   s=q=0
   for x in range(sw):
    a=pix[y*sw+x];s+=a;q+=a*a;sums.append(s+(sums[(y-1)*sw+x]if y else 0));sqs.append(q+(sqs[(y-1)*sw+x]if y else 0))
  off=[]
  for node in range(2913):
   for k in range(3):
    x,y,rw,rh=rects[node*12+k*4:node*12+k*4+4]
    off.extend([-1]*4 if k==2 and not(x or y or rw or rh)else[y*sw+x,y*sw+x+rw,(y+rh)*sw+x,(y+rh)*sw+x+rw])
  trace.extend([sw,sh]);vec(pix);vec(sums);vec(sqs);vec(off)
  codes=[];feat=0
  for x in range(sw-23):
   for y in range(sh-23):
    # Source corner differences integrate pixels x+1..x+23, y+1..y+23.
    vals=[pix[yy*sw+xx]for yy in range(y+1,y+24)for xx in range(x+1,x+24)]
    mean=sum(vals);var=(sum(v*v for v in vals)*576-mean*mean)&0xffffffff;norm=math.isqrt(var)if var else 1
    # Exact oracle memoization: all weak rules share the two contributing rectangles;
    # optional third rectangles have zero weight. Source still executes all2913 nodes.
    assert all(rects[i*12:i*12+8]==rects[:8] and weights[i*3:i*3+3]==weights[:3] and trees[i]==trees[0] and a1[i]==a1[0] and a2[i]==a2[0] for i in range(2913))
    v=0
    for k in range(2):
     rx,ry,rw,rh=rects[k*4:k*4+4]
     area=sum(pix[(y+ry+yy)*sw+x+rx+xx]for yy in range(1,rh+1)for xx in range(1,rw+1))
     v+=area*weights[k]
    weak=a2[0]if v>=trees[0]*norm else a1[0];code=1
    for stage,nn in enumerate(stages):
     ss=nn*weak;feat+=nn
     if ss<.4*sth[stage]:code=-stage;break
    codes.extend([x,y,code])
    if code>0:candidates.append((rnd(f32(x*factor)),rnd(f32(y*factor)),ww,ww))
  vec(codes);vec(flat(candidates));scales.append({'w':sw,'h':sh,'windows':len(codes)//3,'positive':sum(codes[i]>0 for i in range(2,len(codes),3)),'weak_evaluations':feat});factor=f32(factor*f32(1.2))
 grouped=group(candidates);vec(flat(grouped));draw=list(img)
 for x,y,rw,rh in grouped:
  for i in range(rw):draw[w*y+x+i]=255
  for i in range(rh):draw[w*(y+i)+x+rw]=255
  for i in range(rw):draw[w*(y+rh)+x+rw-i]=255
  for i in range(rh):draw[w*(y+rh-i)+x]=255
 vec(draw);return trace,grouped,scales

def main():
 stages=[117]*13+[116]*12;meta=[];blob=bytearray(struct.pack('<I',3));totalout=0;prefix=0
 for w,h,mode in[(32,32,0),(48,40,1),(65,49,2)]:
  img=[(x*7+y*3+(x//8%2)*50)%192 for y in range(h)for x in range(w)]
  rects=[];weights=[]
  for i in range(2913):
   rects += [0,0,11,23,11,0,12,23]+([0,0,0,0]if i%2 else[3,3,8,8]);weights += [-1,1,0]
  trees=[0]*2913;a1=[1 if mode==0 else-1]*2913;a2=[1]*2913;sth=[0]*25
  if mode==1:sth[0]=100000
  elif mode==2:sth[1:]=[-100000]*24
  expected,grouped,scales=oracle(w,h,img,mode,rects,stages,weights,trees,a1,a2,sth)
  inputs=img+stages+rects+weights+trees+a1+a2+sth
  raw=struct.pack('<'+'q'*len(expected),*expected);z=zlib.compress(raw,9)
  blob+=struct.pack('<6I',w,h,mode,len(inputs),len(expected),len(z))+struct.pack('<'+'i'*len(inputs),*inputs)+z
  meta.append({'width':w,'height':h,'classifier_mode':mode,'stages':25,'nodes':2913,'input_values':len(inputs),'trace_values':len(expected),'grouped_rectangles':grouped,'scales':scales});prefix+=len(inputs)+len(rects);totalout+=len(expected)
 (P/'fixtures.bin').write_bytes(blob);(P/'fixture_manifest.json').write_text(json.dumps({'cases':meta,'input_prefix_count':prefix,'trace_values':totalout,'fixture_bytes':len(blob),'fixture_sha256':hashlib.sha256(blob).hexdigest(),'oracle':'Independent direct pixel-patch rectangle sums, integer sqrt/UInt32 normalization and connected-component grouping; synthetic classifier not trained face model.'},indent=2)+'\n');print(json.dumps({'cases':meta,'prefix':prefix,'trace':totalout}))
if __name__=='__main__':main()
