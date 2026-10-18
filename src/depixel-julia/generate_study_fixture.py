"""CPU-only image, quantization-margin and abstract-graph decision oracle."""
from pathlib import Path
import struct,math,json,hashlib,random
p=Path(__file__).resolve().parent
F32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
coef=[F32(x) for x in [.299,.587,.114,.713,.564]]
def color(rgb):
 r,g,b=rgb;y=math.fsum(c*x for c,x in zip(coef[:3],rgb));u=coef[3]*(r-y)+.5;v=coef[4]*(b-y)+.5;channels=[y,u,v]
 margin=min(min(t*255-math.floor(t*255),math.ceil(t*255)-t*255) for t in channels);assert margin>.01
 assert all(.001<t<.999 for t in channels)
 return tuple(math.floor(t*255) for t in channels),margin
palette=[(.25,)*3,(.75,)*3,(.375,)*3,(.125,.25,.375),(.375,.125,.25),(.75,.375,.125),(.25,.625,.75),(.5,)*3,(.875,.75,.5)]
for v in palette:color(v)
steps=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
def similar(a,b):return all(abs(x-y)<=t for x,y,t in zip(a,b,[48,7,6]))
def compute(w,h,img):
 colors=[color(x)[0] for x in img];adj=[];tmp=[]
 for r in range(h):
  for c in range(w):
   node=r*w+c;neighbors=[];bits=0
   for bit,(dr,dc) in enumerate(steps):
    rr,cc=r+dr,c+dc
    if 0<=rr<h and 0<=cc<w and similar(colors[node],colors[rr*w+cc]):bits|=1<<bit;neighbors.append(rr*w+cc)
   adj.append(neighbors);y,u,v=colors[node];tmp.append((v<<24)|(u<<16)|(y<<8)|bits)
 for node,neighbors in enumerate(adj):
  for neighbor in neighbors:assert node in adj[neighbor]
 walkstats={'steps':0,'cycles_capped':0,'outgoing_downleft_seen':0};decisions={'crosses':0,'fully_connected':0,'left':0,'right':0,'ties_both_retained':0,'island_left':0,'island_right':0}
 def score(a,b):
  count=1
  for start,incoming in [(a,b),(b,a)]:
   current=start;previous=incoming
   while len(adj[current])==2 and count<w*h:
    assert previous in adj[current];nxt=next(q for q in adj[current] if q!=previous)
    if (nxt//w-current//w,nxt%w-current%w)==(1,-1):walkstats['outgoing_downleft_seen']+=1
    previous,current=current,nxt;count+=1;walkstats['steps']+=1
   if len(adj[current])==2 and count==w*h:walkstats['cycles_capped']+=1
  return count
 out=[]
 for r in range(h):
  for c in range(w):
   i=r*w+c;bits=0
   if r<h-1 and c<w-1:
    right=i+1;below=i+w;br=i+w+1
    bits=int(right in adj[i]) | (int(right in adj[br])<<1) | (int(below in adj[br])<<2) | (int(below in adj[i])<<3)
    dl=br in adj[i];dr=below in adj[right]
    if dl and dr:
     decisions['crosses']+=1
     if (right in adj[i] or below in adj[i]) and (i in adj[right] or br in adj[right]):decisions['fully_connected']+=1;out.append((tmp[i]&0xffffff00)|bits);continue
     wl=5 if len(adj[i])==1 and br in adj[i] else 0;wr=5 if len(adj[right])==1 and below in adj[right] else 0;decisions['island_left']+=int(wl>0);decisions['island_right']+=int(wr>0)
     sl=sr=0
     for rr in range(max(0,r-3),min(h-1,r+4)+1):
      for cc in range(max(0,c-3),min(w-1,c+4)+1):
       j=rr*w+cc
       if j not in [i,right]:sl+=similar(colors[i],colors[j]);sr+=similar(colors[right],colors[j])
     wr+=max(sl-sr,0);wl+=max(sr-sl,0)
     sl=score(i,br);sr=score(right,below);wl+=max(sl-sr,0);wr+=max(sr-sl,0)
     if wl>wr:bits|=16;decisions['left']+=1
     elif wr>wl:bits|=32;decisions['right']+=1
     else:bits|=16|32;decisions['ties_both_retained']+=1
    else:bits|=int(dl)<<4;bits|=int(dr)<<5
   out.append((tmp[i]&0xffffff00)|bits)
 return tmp,out,{'walk':walkstats,'decisions':decisions}
cases=[]
for w,h,kind in [(32,16,'flat'),(32,16,'checker'),(16,32,'checker'),(16,16,'curve'),(16,16,'island'),(32,32,'palette'),(16,16,'chain')]:
 image=[palette[1]]*(w*h)
 if kind=='flat':image=[palette[2]]*(w*h)
 elif kind=='checker':image=[palette[(r+c)%2] for r in range(h) for c in range(w)]
 elif kind=='curve':
  for r,c in [(5,4),(4,5),(3,6),(2,6),(1,5),(1,4),(2,3),(3,2)]:image[r*w+c]=palette[0]
 elif kind=='island':
  for r,c in [(5,5),(6,6)]:image[r*w+c]=palette[0]
 elif kind=='palette':
  rng=random.Random(19937);image=[palette[rng.randrange(len(palette))] for _ in image]
 elif kind=='chain':
  for r,c in [(3,2),(4,3),(5,4),(6,4),(7,3),(7,2),(6,1),(5,0)]:image[r*w+c]=palette[0]
 tmp,out,stats=compute(w,h,image);cases.append((w,h,kind,image,tmp,out,stats))
words=[]
for w,h,kind,img,*_ in cases:
 for pixel in img:words.extend(struct.unpack('<3I',struct.pack('<3f',*pixel)))
for w,h,*_ in cases:words.extend([w,h])
raw=struct.pack('<'+'I'*len(words),*words);(p/'fixture.bin').write_bytes(raw);gold=[]
for w,h,kind,img,tmp,out,stats in cases:
 gold.extend([0xfedcba98]*8+tmp+[0xfedcba98]*8);gold.extend([0xfedcba98]*8+out+[0xfedcba98]*8)
 rgb=[F32(-12345.)]*24+[x for pixel in img for x in pixel]+[F32(-12345.)]*24;gold.extend(struct.unpack('<'+'I'*len(rgb),struct.pack('<'+'f'*len(rgb),*rgb)))
(p/'oracle.bin').write_bytes(struct.pack('<'+'I'*len(gold),*gold));margin=min(color(v)[1] for v in palette);bound=32*2**-23*255
meta={'cases':[{'w':w,'h':h,'kind':kind,'pixel_count':w*h,**stats} for w,h,kind,img,tmp,out,stats in cases],'palette':palette,'source_Float32_coefficients':coef,'quantization_margin_min_255scaled':margin,'conservative_32eps_Float32_times255_operation_error_allowance':bound,'margin_exceeds_bound':margin>bound,'quantization_contract':'All palette YUV channels strictlyinside(0,1), >.01 fromnearestintegerafter*255; bound32eps*255 <.001 ensures sourceFloat32/FMAassociationcannotchangebins withinselectedfixture. Exactquantizedoutputs, notintegeroutputtolerance. No arbitraryRGB/thresholdboundaryclaim.','prefix_count':len(words),'output_count':len(gold),'artifact_count':len(words)+len(gold),'input_sha256':hashlib.sha256(raw).hexdigest(),'oracle_sha256':hashlib.sha256((p/'oracle.bin').read_bytes()).hexdigest(),'algorithm':'Independent explicit undirected adjacency lists, geometric neighbor traversal with reciprocity, local color-frequency/degree1/cappedcurve-step scores, sourcefullyconnectedshortcut andequalweighttiekeepingbothdiagonals preserved. Notfullplanarvectorization/imagequalityoracle.','corrected_native_scope':'Rectangularrow/columnbounds, missingcurve64break, low8bitislandtests; independentlywitnessedbeforeGPU. Original preserved, no invalidGPUexecution.'}
(p/'fixture_description.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'prefix_count':len(words),'output_count':len(gold),'artifact_count':len(words)+len(gold),'margin':margin,'allowance':bound,'case_stats':meta['cases']},indent=2))
