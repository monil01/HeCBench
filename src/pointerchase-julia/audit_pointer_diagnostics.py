"""Read-only full raw pointer graph and cursor checks, separate from primary output."""
import hashlib,json,struct,sys
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
N=262144
for model in ['cuda','cuda.jl','ka','jacc']:
 directory=root.parents[1]/'.porting-state/porting_logs'/f'pointerchase-julia-{model}'/'diagnostics'
 for trial in range(1,6):
  p=directory/f'trial{trial}_raw_graph_cursor.bin';raw=p.read_bytes()
  assert len(raw)==8*(1+N+4)
  words=struct.unpack('<'+str(1+N+4)+'Q',raw);base=words[0]
  graph=all(v==base+8*((i+16)%N) for i,v in enumerate(words[1:1+N]))
  offsets=[v-base for v in words[-4:]]
  expected=[8*((a*16)%N) for a in [0,1,17,1000000]]
  passed=graph and offsets==expected and base%8==0 and all(v!=0 for v in words[-4:])
  assert passed,(model,trial)
  rows.append(dict(model=model,trial=trial,path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),base_address=base,nodes=N,graph_checked=True,cursor_counts=[0,1,17,1000000],cursor_offsets_bytes=offsets,pass_=passed))
for row in rows:row['pass']=row.pop('pass_')
result=dict(all_pass=True,rows=rows,primary_numerical_agreement_claim=False,scope='Auxiliary untimed full raw graph and final cursor checks. Process-local raw pointers retained, logical graph checked independently, no primary numerical store fabricated.',active_cycle_nodes=N//16,accesses_per_selected_block=1000000,selected_block_ids='All native device SMcount block IDs; no proof of physical SM affinity')
(root/'pointer_diagnostic_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'all_pass':True,'primary_numerical_agreement_claim':False}))
