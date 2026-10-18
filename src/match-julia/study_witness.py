"""Independent integer Gram-matrix oracle; Python standard library only."""
import json,sys,hashlib,struct
from pathlib import Path
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());expect=[];prefix=[];stats=[]
for c in range(2):
 A=[[((i*13+d*7)%17-8)if c==0 else(0 if i==0 else 16 if d==i else 0)for d in range(128)]for i in range(128)]
 B=[[((i*19+d*11)%23-11)if c==0 else(16 if i in(0,127)else-16)for d in range(128)]for i in range(128)]
 prefix.extend(x/16 for row in A for x in row);prefix.extend(x/16 for row in B for x in row);prefix.extend([128,128,3]);dots=[[sum(x*y for x,y in zip(a,b))for b in B]for a in A]
 for k in range(1,11):
  scores=[];indices=[]
  for row in dots:
   best=max(0,max(row));ties=[j for j,x in enumerate(row)if x==best]
   if best<=0:idx=-1
   elif k<=5:idx=min(ties)
   else:
    tile=16 if k==6 else 32
    idx=min(ties,key=lambda j:((j%tile)//4,j//tile,j%4))
   scores.append(best/256);indices.append(idx)
  expect.extend(scores);expect.extend(indices)
 stats.append({'fixture':c,'all128x128_exact_integer_dot_products':16384,'zero_floor_minus_one_rows':sum(max(row)<=0 for row in dots),'rows_with_positive_argmax_ties':sum(row.count(max(row))>1 and max(row)>0 for row in dots),'max_abs_integer_dot':max(abs(x)for row in dots for x in row)})
expected=tuple(prefix+expect);assert len(expected)==70662;reports=[]
for model,m in r['models'].items():
 for t in m['trials']:
  path=Path(t['output_artifact']['saved_path']);raw=path.read_bytes();got=struct.unpack('<70662d',raw);assert got==expected,(model,t['trial'],[(i,x,y)for i,(x,y)in enumerate(zip(got,expected))if x!=y][:10]);reports.append({'model':model,'trial':t['trial'],'all_input_control_values':65542,'all_scores_indices':5120,'exact_match':True,'sha256':hashlib.sha256(raw).hexdigest()})
w={'benchmark':'match','result':str(rp),'captured_trials_checked':len(reports),'method':'Independent arbitrary-precision integer matrix product of exact dyadic numerators; mathematical max and minimum native-priority key among ties. No shared reduction code. Scores divided by256 exactly.','fixtures':stats,'scope':'Complete scores and indices; pairwise intermediate dot matrix is independent oracle only, not exported GPU output. 128points128dimensions/all10variants. Native max floor0 yields score0/index-1 for all-nonpositive rows, not unrestricted signed argmax.','reports':reports};path=Path('.porting-state/portable_julia/scientific_witnesses_300/match.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(w,indent=2)+'\n');print(json.dumps(w))
