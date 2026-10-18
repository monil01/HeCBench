from pathlib import Path
from array import array
import json,hashlib,sys,math
r=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;run=sys.argv[1];bp=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bp/'result.json').read_text());pre=array('d',(p/'fixture.bin').read_bytes());gold=array('d',(p/'oracle.bin').read_bytes());kind=(p/'kinds.bin').read_bytes();rows=[];hashes=set();worst={str(k):dict(max_abs=0,max_rel_nonzero=0,max_abs_zero=0,max_budget_fraction=0)for k in range(3)}
segments=[];cursor=0
def seg(name,n,typ='general'):
 global cursor
 cursor+=64;segments.append((cursor,cursor+n,name,typ));cursor+=n+64
def capture(case,stage,stats):
 for label in ['x1','y1','z1','x2','y2','z2','psi']:seg(label,57344,'psi'if label=='psi'else'coordinates')
 if stats:
  global cursor
  cursor+=64
  for label in ['r1','r2','r12','accept']:segments.append((cursor,cursor+57344,label,'counts'if label=='accept'else'distance_accumulators'));cursor+=57344
  cursor+=64
 seg('RNG',57344,'RNG')
for ci in range(2):
 capture(ci,'initial',False);capture(ci,'burn',True)
 for sample in range(2):capture(ci,sample,True);seg('four_totals',4,'totals');seg('last_accept_partial',224,'counts')
assert cursor==len(gold)
field_errors={};checked_hashes=set()
for m,z in d['models'].items():
 for t in z['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();v=array('d',raw);assert v[:9]==pre and len(v)==len(gold)+9;sha=hashlib.sha256(raw).hexdigest();hashes.add(sha);assert len(t['sample_times_seconds'])==4 and abs(sum(t['sample_times_seconds'])-t['time_seconds'])<1e-12
  if sha not in checked_hashes:
   checked_hashes.add(sha)
   for start,end,name,typ in segments:
    w=field_errors.setdefault(typ,dict(max_abs=0,max_rel_nonzero=0,max_abs_zero=0))
    for x,y in zip(v[9+start:9+end],gold[start:end]):
     e=abs(x-y);w['max_abs']=max(w['max_abs'],e)
     if y:w['max_rel_nonzero']=max(w['max_rel_nonzero'],e/abs(y))
     else:w['max_abs_zero']=max(w['max_abs_zero'],e)
  if sha in {x['sha256']for x in rows}:
   rows.append(dict(model=m,trial=t['trial'],sha256=sha,path=str(f),words=len(v),exact_RNG_acceptcounts_guard_words_checked=True,comparison='reused_after_full_file_rehash_identical_bytes'));continue
  for k,x,y in zip(kind,v[9:],gold):
   assert math.isfinite(x);e=abs(x-y);w=worst[str(k)];w['max_abs']=max(w['max_abs'],e)
   if y:w['max_rel_nonzero']=max(w['max_rel_nonzero'],e/abs(y))
   else:w['max_abs_zero']=max(w['max_abs_zero'],e)
   if k==2:assert x==y
   else:
    budget=(1e-12+2e-5*abs(y))if k==1 else(2e-5+3e-5*abs(y));w['max_budget_fraction']=max(w['max_budget_fraction'],e/budget);assert e<=budget
  assert len(t['sample_times_seconds'])==4 and abs(sum(t['sample_times_seconds'])-t['time_seconds'])<1e-12
  rows.append(dict(model=m,trial=t['trial'],sha256=sha,path=str(f),words=len(v),exact_RNG_acceptcounts_guard_words_checked=True))
assert len(rows)==20 and d['matched_performance_valid']
a=dict(run=run,all_pass=True,full20_artifacts=len(rows),words_per_artifact=5057177,unique_hashes=sorted(hashes),actual_field_metrics=field_errors,per_field_errors={'coordinates_distances_sums':worst['0'],'wavefunction':worst['1'],'RNG_acceptance_counts_guards':worst['2']},all4_source_population_57344=True,short_named_chains_not_equilibrium=True,local_energy_not_in_source=True,oracle='IndependentFloat64hypot/analyticwavefunction/LCG/MCMC andmathematicalgroupsum vsFP32source; margin tableinfixture_description, exactdecisioncounts/RNG checked; no formalglobalerrorbound.',rows=rows)
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:v for k,v in a.items()if k!='rows'}))
