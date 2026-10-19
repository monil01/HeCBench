#!/usr/bin/env python3
"""Read preserved actual timed/replay buffers; prove concrete conflicting source owners."""
from pathlib import Path
import struct,json,hashlib
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;raw=R/'.porting-state/portable_julia/graphB+_raw_v1'
a=raw/'cuda_jl_1875536_trial1_n65_timed_with_replay_dfs.canonical.i64';b=raw/'cuda_jl_1875536_trial1_n65_it2.raw.i64.canonical.i64';physical=raw/'cuda_jl_1875536_trial1_n65_timed.raw.i64'
def rd(p):z=p.read_bytes();return struct.unpack('<'+'q'*(len(z)//8),z)
x,y,z=rd(a),rd(b),rd(physical);c=json.loads((P/'fixtures.json').read_text())['cases'][1];n=c['n'];e=len(c['nlist']);witness=[]
for u in range(n):
 for j in range(c['nindex'][u],c['nindex'][u+1]):
  v=c['nlist'][j];off=5*n+6*j;got=x[off+5];expected=y[off+5]
  if got!=expected:
   locations=[q for q in range(c['nindex'][u],c['nindex'][u+1])if z[q]>>1==v];assert len(locations)==1;q=locations[0];assert z[q]&1 and z[e+2*q+1]&1==0 and got==1 and expected==0
   assert q//256!=u//256
   witness.append(dict(source=u,neighbor=v,canonical_edge_index=j,physical_edge_index=q,clear_writer=dict(thread=q%256,block=q//256,value=True),tree_writer=dict(thread=u%256,block=u//256,value=False),actual_timed_minus=got,replay_minus=expected,source_CLI_directed_edge_printed=u<v))
assert len(witness)==19
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(status='observed_faithful_target_timed_replay_divergence_and_original_source_cross_block_write_conflict',raw_sources={str(p.relative_to(R)):sha(p)for p in [a,b,physical]},original_native_source_sha256=sha(R/'src/graphB+-cuda/kernels.h'),original_native_five_trials='all_passed; no observed native wrong-output claim',conflicting_source_statements=['kernels.h initMinus lines652-654 minus[j]=true','kernels.h initMinus lines657-660 minus[j]=einfo[j].end&1'],legal_schedules=['edge block1 true then node block0 false ->false','node block0 false then edge block1 true ->true'],source_comments_intend='default all edge signs true, copy original sign for both directions of all tree edges',affected_edge_count=len(witness),affected_source_CLI_printed_edges=sum(w['source_CLI_directed_edge_printed']for w in witness),witness=witness,correction='single per-edge owner sets tree sign from same edge flag/info, otherwise true; same one launch; no operator/tolerance relaxation')
(P/'initialization_race_witness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['status','affected_edge_count','affected_source_CLI_printed_edges']}))
