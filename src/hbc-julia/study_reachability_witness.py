#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,sys
P=Path(__file__).resolve().parent;root=P.parents[1];assert sys.byteorder=='little'
a=memoryview((P/'fixtures.bin').read_bytes()).cast('i');q=1;cs=[]
for _ in range(a[0]):
 n,m=a[q:q+2];q+=2;R=a[q:q+n+1];q+=n+1;C=a[q:q+2*m];q+=2*m;F=a[q:q+2*m];q+=2*m;cs.append((n,m,R,C,F))
n,m,R,C,F=cs[-1];seen=set();components=[]
for source in range(n):
 if source in seen:continue
 stack=[source];seen.add(source);members=[]
 while stack:
  v=stack.pop();members.append(v)
  for w in C[R[v]:R[v+1]]:
   if w not in seen:seen.add(w);stack.append(w)
 components.append(members)
layout_path=next((root/'.porting-state/portable_julia/hbc_raw_full_v1').glob('native_*_trial1_case3.layout.json'));layout=json.loads(layout_path.read_text());raw_path=Path(str(layout_path).replace('.layout.json','.raw.bin'));raw=raw_path.read_bytes();by={x['name']:x for x in layout['buffers']}
def row(name,block):
 b=by[name];width=n+1 if name=='ep'else n;z=memoryview(raw)[b['offset_bytes']:b['offset_bytes']+b['count_elements']*b['element_bytes']].cast('i');start=(block+1)*b['stride_elements'];return z[start:start+width]
rows=[]
for block in range(80):
 d=row('d',block);S=row('S',block);ep=row('ep',block);roots=[v for v in range(n)if d[v]==0]
 if len(roots)!=1:continue
 source=roots[0];reachable=[v for v in range(n)if d[v]!=2147483647];depth=max(d[v]for v in reachable);k=len(reachable)
 assert sorted(S[:k])==reachable;assert ep[depth+1]==k
 if k<n:rows.append(dict(block=block,source=source,reachable=k,total=n,depth=depth,actual_final_endpoint=ep[depth+1],S_reachable_prefix_exact=True,unread_S_tail_elements=n-k))
r=dict(status='auxiliary_checker_connectedness_assumption_refuted_by_source_input_and_retained_actual_workspace',component_sizes=sorted(map(len,components),reverse=True),components=len(components),raw_file=str(raw_path.relative_to(root)),raw_sha256=hashlib.sha256(raw).hexdigest(),rows_with_partial_reachability=rows,source_Brandes_contract='Only BFS reachable vertices are in S/endpoints; unreachable d=INT_MAX/sigma=0/delta=0. Unread S tail remains stale, fullbytesretained.',BC_oracle_budget_unchanged=True,device_body_unchanged=True);(P/'workspace_reachability_witness.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items()if k not in['rows_with_partial_reachability']});print('partial_rows',len(rows),rows[:2])
