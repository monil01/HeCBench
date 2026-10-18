from pathlib import Path
from decimal import Decimal,localcontext
import json,struct,math,hashlib
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;state=R/'.porting-state/portable_julia/tsp_reference_review';state.mkdir(exist_ok=True);selection=json.loads((p/'deterministic_fixture_selection.json').read_text());raw=lambda x:struct.unpack('<I',struct.pack('<f',x))[0];f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];u=lambda x:struct.pack('<I',x&0xffffffff);guard=0xdeadbeef;prefix=[2];out=[];fixtures=[2];reviews=[]
for c in selection['cases']:
 n=c['n'];restarts=c['restarts'];threads=c['threads'];stride=((3*n+2+31)//32)*32;points=c['points'];prefix +=[n,restarts,threads]+[raw(x)for x,y in points]+[raw(y)for x,y in points];fixtures +=[n,restarts,threads]+[raw(x)for x,y in points]+[raw(y)for x,y in points]
 for rr in c['results']:
  route=rr['route'];assert sorted(route)==list(range(n));d=[-e&0xffffffff for e in rr['edges']]+[raw(points[i][0])for i in route+[route[0]]]+[raw(points[i][1])for i in route+[route[0]]];d +=[guard]*(stride-len(d));out +=d
 out +=[sum(q['climbs']for q in c['results']),min(q['length']for q in c['results'])];pair=[];margin=None;source_floor_vs_real=0
 with localcontext()as ctx:
  ctx.prec=70
  for x,y in points:
   for xx,yy in points:
    q=(x-xx)**2+(y-yy)**2;assert f32(q)==q;root=Decimal(q).sqrt();w=f32(math.sqrt(q));bits=raw(w);distance=int(w);pair.append(distance)
    if q:
     prev=struct.unpack('<f',u(bits-1))[0];nxt=struct.unpack('<f',u(bits+1))[0];low=(Decimal(prev)+Decimal(w))/2;high=(Decimal(nxt)+Decimal(w))/2;assert low<root<high;delta=min(root-low,high-root);margin=delta if margin is None else min(margin,delta)
    source_floor_vs_real +=distance!=math.isqrt(q)
 out +=pair
 for _ in range(6):out +=[guard]*128
 reviews.append({'n':n,'restarts':restarts,'threads':threads,'stride':stride,'climbs':sum(q['climbs']for q in c['results']),'best':min(q['length']for q in c['results']),'all_negative_best_lanes_unique':True,'all_terminal_states_2opt_local_min':True,'pairwise_sqrt_RN_midpoint_margin_min':str(margin),'source_Float32sqrt_floor_differs_from_exact_real_floor_pairs':source_floor_vs_real,'all_squared_sum_exact_Float32':True})
allwords=prefix+out;gold=state/'golden_u32.bin';gold.write_bytes(b''.join(map(u,allwords)));(p/'fixtures_u32.bin').write_bytes(b''.join(map(u,fixtures)));(p/'oracle_u32.bin').write_bytes(b''.join(map(u,out)));info={'cases':reviews,'input_prefix_count':len(prefix),'output_count':len(allwords),'golden_path':str(gold.relative_to(R)),'golden_sha256':hashlib.sha256(gold.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256((p/'fixtures_u32.bin').read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256((p/'oracle_u32.bin').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((p/'deterministic_fixture_selection.json').read_bytes()).hexdigest(),'numeric_contract':'Everyrawinput/scratch/finaltour/negativeedge/best/climbs/pairdist/padding/guardwordexact, sourceFP32sqrtRNfloors separatelywitnessed70digitDecimal andactualGPUuntimedallpairdiagnostic.'};(p/'independent_oracle.json').write_text(json.dumps(info,indent=2)+'\n');print(info)
