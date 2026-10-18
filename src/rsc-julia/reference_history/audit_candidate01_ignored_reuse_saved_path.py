"""Read-only replay: bind every canonical artifact and full raw append sequence."""
from pathlib import Path
import struct,json,hashlib,sys
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];bundle=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bundle/'result.json').read_text());fd=json.loads((p/'fixture_description.json').read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f,n):
 b=Path(f).read_bytes();assert len(b)==8*n,(f,len(b),n);return list(struct.unpack('<'+str(n)+'d',b))
pre=load(p/'fixture.bin',3061);gold=load(p/'oracle.bin',6756);allrows=[];orders=[set() for _ in range(4)];bestids=[set() for _ in range(4)];canonicalhashes=set()
for model,v in d['models'].items():
 assert v['complete'] and len(v['trials'])==5
 for trial,row in enumerate(v['trials'],1):
  # Runner's archive path is independent of the temporary path emitted on stdout.
  f=bundle/f'{model}_trial{trial}_output_trial{trial}.bin';actual=load(f,9817);assert actual[:3061]==pre and actual[3061:]==gold
  canonicalhashes.add(sha(f));rawspec=row['raw_artifact'];rf=Path(rawspec['path']);raw=load(rf,6760);pos=gp=0;cases=[]
  for ci,case in enumerate(fd['cases']):
   n,m=case['n'],case['models'];count,best,minimum,valid=map(int,raw[pos:pos+4]);pos+=4;assert count==case['accepted'] and minimum==case['minimum_outliers'] and valid==1
   params=raw[pos:pos+4*m];pos+=4*m;assert params==gold[gp+3:gp+3+4*m]
   pairs=[tuple(map(int,raw[pos+2*i:pos+2*i+2])) for i in range(m)];pos+=2*m
   active=pairs[:count];assert all(a==(0,0) for a in pairs[count:]);assert len({a for a,b in active})==count
   expected=[tuple(map(int,gold[gp+3+4*m+2*i:gp+3+4*m+2*i+2])) for i in range(count)]
   assert sorted(active)==expected
   strictmin=n;strictbest=-1
   for a,b in active:
    if b<strictmin:strictmin=b;strictbest=a
   assert (minimum,best)==(strictmin,strictbest)
   assert (count==0 and best==-1) or best in case['minimizer_ids']
   tail=raw[pos:pos+4*n+4*m+80];pos+=len(tail)
   assert tail==gold[gp+3+6*m:gp+case['artifact_values']]
   assert tail[-80:]==[16384]*80
   orders[ci].add(tuple(active));bestids[ci].add(best);gp+=case['artifact_values']
   cases.append(dict(case=ci+1,candidate_count=count,best_id=best,minimum_outliers=minimum,raw_candidate_order_sha256=hashlib.sha256(struct.pack('<'+str(2*count)+'i',*[x for a in active for x in a])).hexdigest(),all_source_strict_best_selection_and_minimizer_membership=True))
  assert pos==len(raw) and gp==len(gold)
  allrows.append(dict(model=model,trial=trial,canonical_artifact=dict(path=str(f.relative_to(r)),sha256=sha(f),bytes=f.stat().st_size),raw_artifact=dict(path=str(rf.resolve().relative_to(r)),sha256=sha(rf),bytes=rf.stat().st_size,count=len(raw)),cases=cases,all_pass=True))
out=dict(benchmark='rsc',configuration_id=d['configuration_id'],result_sha256=sha(bundle/'result.json'),configuration_sha256=sha(bundle/'config.json'),all20_checked=True,all_pass=True,canonical_artifacts_bitwise_identical=len(canonicalhashes)==1,canonical_artifact_values=9817,input_prefix_values=3061,full_raw_artifact_values=6760,numerical_agreement_claim=False,raw_output_identity_claim=False,numerical_agreement_kind='property_equivalence_not_identical_raw_values',case_distinct_raw_orders=[len(s) for s in orders],observed_best_ids=[sorted(s) for s in bestids],raw_append_order_nondeterminism_observed=any(len(s)>1 for s in orders),independent_oracle='Exact Fraction solved linear-system parameters plus actual source Float32 residual/truncation convention, all accepted IDs/counts and minimum-membership; rejected-model numerical counts not stored by native and not claimed captured.',all20_full_raw_arrays_and_guards_retained=True,rows=allrows)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='rows'})
