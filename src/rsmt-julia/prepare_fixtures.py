#!/usr/bin/env python3
"""Parse exact original routing input; preserve source coordinate projection/truncation."""
from pathlib import Path
import argparse,json,hashlib,struct,random
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();src=Path(a.source);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
assert src.stat().st_size==194316711
assert hashlib.md5(src.read_bytes()).hexdigest()=='8b70aae8ccc5aa050941b5afb6854b40'
selected=[];bydegree={};hist={};allpins=0
with src.open()as f:
 grid=list(map(int,f.readline().split()[1:]));[f.readline()for _ in range(5)];llx,lly,tw,th=map(int,f.readline().split());line=f.readline()
 while not line.strip():line=f.readline()
 assert line.split()[:2]==['num','net'];nn=int(line.split()[2])
 for i in range(nn):
  name,ident,degree,width=f.readline().split();degree=int(degree);points=[]
  for j in range(degree):
   x,y,z=map(int,f.readline().split());assert 0<z<=grid[2]
   if not(llx<=x<grid[0]*tw):x=grid[0]*tw-1
   if not(lly<=y<grid[1]*th):y=grid[1]*th-1
   assert 0<=x<2097151 and 0<=y<2097151
   if j<256:points.append((x,y))
  d=min(degree,256);hist[d]=hist.get(d,0)+1;allpins+=d
  row={'source_index':i,'name':name,'original_degree':degree,'points':points}
  if i<8192:selected.append(row)
  if d not in bydegree:bydegree[d]=row
extra=[v for d,v in sorted(bydegree.items())if v['source_index']>=8192]
cases=[('original_first8192_plus_degree_coverage',selected+extra)]
rng=random.Random(53732026);controls=[]
for d in [2,3,4,5,17,31,32,33,63,64,127,128,255,256]:
 controls.append({'source_index':None,'name':'independent_random_degree'+str(d),'original_degree':d,'points':[(rng.randrange(0,4096),rng.randrange(0,4096))for _ in range(d)]})
for pts in [[(0,0),(7,11)],[(0,0),(8,0),(3,9)],[(3,3)]*3,[(0,0),(0,8),(8,0),(8,8)],[(i,0)for i in range(33)]]:
 controls.append({'source_index':None,'name':'algebraic_control'+str(len(controls)),'original_degree':len(pts),'points':pts})
cases.append(('degree_tail_duplicate_collinear_controls',controls))
meta={'source':{'path':str(src),'size':src.stat().st_size,'md5':hashlib.md5(src.read_bytes()).hexdigest(),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'nets':nn,'truncated_total_pins':allpins,'degree_histogram':hist},'cases':[],'scope':'Bounded named source sample plus synthetic controls; full2.6M-net CLI not measured.'}
for name,nets in cases:
 idx=[0];x=[];y=[]
 for net in nets:
  for xx,yy in net['points']:x.append(xx);y.append(yy)
  idx.append(len(x))
 data=struct.pack('<ii',len(nets),len(x))+struct.pack('<'+'i'*(len(idx)+len(x)+len(y)),*(idx+x+y));path=out/(name+'.bin');path.write_bytes(data)
 meta['cases'].append({'name':name,'fixture':path.name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'nets':len(nets),'pins':len(x),'input_degrees':sorted(set(len(v['points'])for v in nets)),'net_sources':[{k:v for k,v in net.items()if k!='points'}for net in nets]})
(out/'fixture_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items()if k!='cases'}));print([(x['name'],x['nets'],x['pins'])for x in meta['cases']])
