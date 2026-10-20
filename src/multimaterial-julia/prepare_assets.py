#!/usr/bin/env python3
"""Deterministic compact multi-material fixtures; stdlib only, binary64 little-endian."""
import array,hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
root.mkdir(parents=True,exist_ok=True)
sx,sy,nm=96,80,50;nc=sx*sy
counts=[1+(i//4+j//4)%4 for j in range(sy) for i in range(sx)]
dense={}
for j in range(sy):
 for i in range(sx):
  c=i+sx*j;k=counts[c];weights=[m+1 for m in range(k)]
  for m in range(k):dense[c,m]=(1+(i%11)*.03125+(j%7)*.015625+m*.125,2+(i%5)*.0625+(j%13)*.03125+m*.25,weights[m]/sum(weights))
D={k:[] for k in ['imaterial','nextfrac','matids','mmc_index','mmc_i','mmc_j','rho_cell','rho_list','vf_list','t_cell','t_list','V','x','y','n']}
D['rho_cell']=[0.]*nc;D['t_cell']=[0.]*nc
D['V']=[1/(sx*sy)]*nc;D['x']=[i/sx for j in range(sy) for i in range(sx)];D['y']=[j/sy for j in range(sy) for i in range(sx)];D['n']=[.75+m*.03125 for m in range(nm)]
mixed=0
for c,k in enumerate(counts):
 if k==1:D['imaterial'].append(1);D['rho_cell'][c]=dense[c,0][0];D['t_cell'][c]=dense[c,0][1]
 else:
  D['imaterial'].append(-mixed);D['mmc_index'].append(len(D['matids']));D['mmc_i'].append(c%sx);D['mmc_j'].append(c//sx);mixed+=1
  for m in range(k):
   r,t,v=dense[c,m];D['matids'].append(m);D['rho_list'].append(r);D['t_list'].append(t);D['vf_list'].append(v);D['nextfrac'].append(len(D['matids']) if m<k-1 else -1)
D['mmc_index'].append(len(D['matids']));ml=len(D['matids'])
O={k:[0.]*n for k,n in [('rho_ave',nc),('p_cell',nc),('p_list',ml),('neighbor_cell',nc),('neighbor_list',ml)]}
fractions={};idx=0
for c,k in enumerate(counts):
 O['rho_ave'][c]=math.fsum(dense[c,m][0]*dense[c,m][2] for m in range(k))/D['V'][c]
 for m in range(k):
  r,t,v=dense[c,m];p=D['n'][m]*r*t/v
  if k==1:O['p_cell'][c]=p
  else:O['p_list'][idx]=p;fractions[c,m]=idx;idx+=1
for j in range(1,sy-1):
 for i in range(1,sx-1):
  c=i+sx*j
  for m in range(counts[c]):
   terms=[]
   for dj in [-1,0,1]:
    for di in [-1,0,1]:
     if di==dj==0:continue
     q=c+di+sx*dj
     if (q,m) in dense:
      dist=(D['x'][c]-D['x'][q])**2+(D['y'][c]-D['y'][q])**2
      terms.append(dense[q,m][0]/dist)
   assert terms,'fixture excludes undefined empty neighbor averages'
   val=math.fsum(terms)/len(terms)
   if counts[c]==1:O['neighbor_cell'][c]=val
   else:O['neighbor_list'][fractions[c,m]]=val
schema={'geometry':[sx,sy,nm],'ncells':nc,'mixed_cells':mixed,'mixed_fractions':ml,'counts_histogram':{str(k):counts.count(k) for k in range(1,5)},'inputs':[],'outputs':[],'guards':17,'guard_value':-987654.25,'oracle':'independently enumerated dense (cell,material) dictionary, math.fsum weighted sums, material membership neighbor lookup; self excluded in corrected contract only','original_defect_witness':'native_diagnostics/original_nonfinite_witness.json','native_reference_repaired':True,'corrected_reference_secondary':True,'tolerance':{'atol':1e-8,'rtol':2e-13},'source_default_file1000x1000_unmeasured':True}
for name,v in D.items():schema['inputs'].append({'name':name,'count':len(v),'integer':name in ['imaterial','nextfrac','matids','mmc_index','mmc_i','mmc_j']})
for name,v in O.items():schema['outputs'].append({'name':name,'count':len(v)})
for mode in [0,1,2]:
 E={k:list(v) for k,v in D.items()}
 if mode==2:
  for c in range(nc):
   if counts[c]>1:E['imaterial'][c]=-D['mmc_index'][-D['imaterial'][c]]
 vals=[float(x) for k in D for x in E[k]]
 a=array.array('d',vals)
 if sys.byteorder!='little':a.byteswap()
 (root/f'input_mode{mode}_f64.bin').write_bytes(a.tobytes())
expected=[]
for k,v in O.items():expected.extend(v+[schema['guard_value']]*schema['guards'])
a=array.array('d',expected)
if sys.byteorder!='little':a.byteswap()
(root/'oracle_f64.bin').write_bytes(a.tobytes())
schema['input_prefix_count']=sum(x['count'] for x in schema['inputs']);schema['output_count']=len(expected)
schema['assets']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.glob('*_f64.bin'))}
(root/'study_plan.json').write_text(json.dumps(schema,indent=2)+'\n')
print(json.dumps({'prefix':schema['input_prefix_count'],'output':len(expected),'mixed':mixed,'fractions':ml,'assets':schema['assets']},indent=2))
