"""Independent CPU spatial tree and warp-group gravitational force/integration.
Geometry partitions round to FP32 to reproduce the declared cell membership;
COM, forces and integration use FP64, independently of CUDA/Julia device code.
"""
import struct,math,json,subprocess,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
CASES=[(256,1),(256,3),(1024,1),(1024,3),(2048,1)]
def solve(flat,n,steps):
 p=[list(flat[4*i:4*i+4])for i in range(n)]
 v=[list(flat[4*n+2*i:4*n+2*i+2])+[flat[6*n+4*i+3]]for i in range(n)]
 acc=[[0.,0.,0.]for i in range(n)];details=[]
 for step in range(steps):
  low=[min(x[d]for x in p)for d in range(3)];high=[max(x[d]for x in p)for d in range(3)]
  radius=f32(max(f32(high[d]-low[d])for d in range(3))*.5)
  center=[f32(f32(low[d]+high[d])*.5)for d in range(3)]
  nodes=[]
  def tree(ids,center,r,depth):
   if len(ids)==1:return ('body',ids[0])
   if depth>=31:raise ValueError('depth beyond native stack')
   ix=len(nodes);nodes.append(None);groups=[[]for _ in range(8)]
   for i in ids:
    octant=sum((1<<d)if center[d]<p[i][d]else 0 for d in range(3));groups[octant].append(i)
   rr=f32(r*.5);children=[]
   for octant,g in enumerate(groups):
    if not g:continue
    c=[f32(center[d]+(rr if octant&(1<<d)else-rr))for d in range(3)]
    children.append(tree(g,c,rr,depth+1))
   mass=math.fsum(p[i][3]for i in ids);com=[math.fsum(p[i][d]*p[i][3]for i in ids)/mass for d in range(3)]
   nodes[ix]={'children':children,'mass':mass,'com':com,'depth':depth,'count':len(ids)}
   return ('cell',ix)
  root=tree(list(range(n)),center,radius,0)
  order=[]
  def emit(ch):
   kind,i=ch
   if kind=='body':order.append(i)
   else:
    for c in nodes[i]['children']:emit(c)
  emit(root);assert sorted(order)==list(range(n))
  newacc=[[0.,0.,0.]for _ in range(n)];visited=0;accepted=0
  for start in range(0,n,32):
   group=order[start:start+32]
   def visit(ch,depth):
    nonlocal visited,accepted
    kind,i=ch;visited+=1
    point=p[i][:3]if kind=='body'else nodes[i]['com'];m=p[i][3]if kind=='body'else nodes[i]['mass']
    diffs=[[point[d]-p[b][d]for d in range(3)]for b in group]
    dist=[math.fsum(x*x for x in diff)+f32(.05*.05)for diff in diffs]
    threshold=(2*radius)**2*4*(.25**depth)+f32(.05*.05)
    if kind=='body' or all(x>=threshold for x in dist):
     accepted+=1
     for b,diff,rr in zip(group,diffs,dist):
      weight=m/(rr*math.sqrt(rr))
      for d in range(3):newacc[b][d]+=diff[d]*weight
    else:
     for c in nodes[i]['children']:visit(c,depth+1)
   for c in nodes[root[1]]['children']:visit(c,0)
  for i in range(n):
   for d in range(3):
    if step>0:v[i][d]+=(newacc[i][d]-acc[i][d])*f32(.0125)
    dh=newacc[i][d]*f32(.0125);vh=v[i][d]+dh
    # Carry public positions rounded as in the GPU while the oracle arithmetic stays FP64.
    p[i][d]=f32(p[i][d]+vh*f32(.025));v[i][d]=vh+dh
  acc=newacc
  details.append({'nodes':len(nodes),'max_depth':max(x['depth']for x in nodes),'warp_node_visits':visited,'accepted_warp_nodes':accepted,'root_mass':nodes[0]['mass']})
 out=[x for row in p for x in row]+[x for row in v for x in row[:2]]+[x for a,w in zip(acc,v)for x in a+[w[2]]]
 return out,details
if __name__=='__main__':
 subprocess.run(['g++','-O2','-I/usr/include','generate_fixture.cpp','-o','/tmp/hecbench_bh_fixture'],cwd=P,check=True)
 metadata=[];expected=[];fixtures=[]
 for n,steps in CASES:
  path=P/f'fixture_{n}.bin'
  if not path.exists():subprocess.run(['/tmp/hecbench_bh_fixture',str(n),str(path)],check=True)
  raw=path.read_bytes();flat=struct.unpack('<'+'f'*(len(raw)//4),raw);assert len(flat)==10*n
  out,desc=solve(flat,n,steps);expected.extend(out);fixtures.extend(flat)
  metadata.append({'n':n,'steps':steps,'fixture':path.name,'fixture_sha256':hashlib.sha256(raw).hexdigest(),'oracle_tree_steps':desc})
 (P/'expected_f64.bin').write_bytes(struct.pack('<'+'d'*len(expected),*expected))
 (P/'fixture_description.json').write_text(json.dumps({'cases':metadata,'input_prefix_count':len(fixtures),'output_count':len(expected),'oracle':'CPU independent spatial partitions/FP64COM+warp-group-opening/force+integration; publicpositions FP32-rounded; no fullapplication orlongtimeaccuracyclaim','atol':1e-5,'rtol':1e-4},indent=2)+'\n')
 print({'cases':len(metadata),'count':len(expected)})
