#!/usr/bin/env python3
"""Independent CPU Petri-net transition phase simulator; exact source untempered MT."""
from pathlib import Path
import json,hashlib,struct
import numpy as np
P=Path(__file__).resolve().parent
def mt_values(seed):
 mt=[seed]
 for i in range(1,624):mt.append((1812433253*(mt[-1]^(mt[-1]>>30))+i)&0xffffffff)
 for i in range(624):
  y=(mt[i]&0x80000000)|(mt[(i+1)%624]&0x7fffffff);mt[i]=mt[(i+397)%624]^(y>>1)^(0x9908b0df if y&1 else 0)
 return mt

def simulate(n,seed):
 count=2*n*n;marks=np.ones(count,dtype='i1');conflict=np.zeros(count,dtype='i4');random=mt_values(seed);waves=[]
 for offset,limit,step in [(0,256,7),(256,256,11),(512,112,13)]:
  entries=[]
  for tid,value in enumerate(random[offset:offset+limit]):
   tr,tc=divmod(value%count,n);above=2*n-1 if tr==0 else tr-1;col=(0 if tc==n-1 else tc+1) if tr&1 else tc;below=0 if tr==2*n-1 else tr+1;a=above*n+col;b=tr*n+tc;c=below*n+col;d=tr*n+(0 if tc==n-1 else tc+1);left_neighbor=below*n+(n-1 if col==0 else col-1);enabled=marks[a]>0 and marks[b]>0
   if enabled:conflict[b]=step
   entries.append((tr,a,b,c,d,left_neighbor,int(marks[a]),int(marks[b]),enabled))
  active=[e for e in entries if e[-1] and ((step&1)==(e[0]&1) or (conflict[e[1]]!=step and conflict[e[5]]!=step))]
  inp=[v for e in active for v in e[1:3]];out=[v for e in active for v in e[3:5]];dup_in=len(inp)-len(set(inp));dup_out=len(out)-len(set(out));unique_tr=len(set(e[2]for e in entries));wave={'step_label':step,'attempts':limit,'unique_random_transitions':unique_tr,'active_threads':len(active),'duplicate_input_write_destinations':dup_in,'duplicate_output_increment_destinations':dup_out,'race_free_individual_byte_writes':dup_in==0 and dup_out==0,'packed_word_RMW_hazard':len(set(v//4 for v in inp))<len(inp) or len(set(v//4 for v in out))<len(out)};waves.append(wave)
  # Only a unique-destination schedule defines the source single-byte write outcome.
  if dup_in or dup_out:return {'n':n,'seed':seed,'waves':waves,'exact_cpu_source_state_defined':False},None,None
  for e in active:marks[e[1]]=e[6]-1;marks[e[2]]=e[7]-1
  for e in active:marks[e[3]]+=1;marks[e[4]]+=1
 unsigned=marks.view('u1');total=int(unsigned.astype('u8').sum());squares=int((unsigned.astype('u8')**2).sum());maximum=int(unsigned.max());variance=float(np.float32(np.float32(squares)/np.float32(count))-np.float32(1))
 assert total==count
 return {'n':n,'seed':seed,'waves':waves,'exact_cpu_source_state_defined':True,'token_total':total,'squared_marking_sum':squares,'max_mark':maximum,'variance_reward_float32':variance},unsigned,conflict
reports=[];outputs=[];conflicts=[]
for seed in range(5489,5493):
 report,mark,conflict=simulate(1024,seed);reports.append(report)
 if mark is not None:outputs.append(mark);conflicts.append(conflict)
small=[simulate(22,seed)[0] for seed in range(5489,5493)]
(P/'diagnostics/transition_schedule_audit.json').write_text(json.dumps({'generator':'IndependentPythonuntemperedMTandbarrierphases','numpy':np.__version__,'primary_n':1024,'primary_steps':312,'trajectories':4,'primary_seeds':list(range(5489,5493)),'primary':reports,'original_N22_first_batch_diagnostic':small,'uninitialized_conflicts_defined_zero_in_named_fixture':True,'full_original_S100000_T1000000_unverified':True},indent=2)+'\n')
if not all(a['exact_cpu_source_state_defined']for a in reports):raise SystemExit('Primary canonical schedule has native byte-write hazards; cannot claim unique numerical oracle')
np.stack(outputs).tofile(P/'study_expected_marks_u8.bin');np.stack(conflicts).astype('<i4').tofile(P/'study_expected_conflicts_i32.bin');np.array([[a['variance_reward_float32'],a['max_mark']]for a in reports],dtype='<f8').tofile(P/'study_expected_rewards_f64.bin');np.array([1024,312,4,5489],dtype='<i4').tofile(P/'study_input_i32.bin')
(P/'fixture_provenance.json').write_text(json.dumps({'generator':'generate_study_fixture.py','generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy':np.__version__,'primary_schedule_proven_byte_race_free':True,'untempered_mt_source_semantics':True,'files':{f.name:{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}for f in P.glob('study_*.bin')}},indent=2)+'\n')
print(json.dumps({'race_free_primary':True,'reports':reports},indent=2))
