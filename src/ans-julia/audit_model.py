"""Full-byte independent serial oracle plus actual GPU scratch/stage-prefix invariants."""
from pathlib import Path
import sys,json,hashlib,numpy as np
P=Path(__file__).resolve().parent;D=Path(sys.argv[1]);DIAG=Path(sys.argv[2]);plan=json.loads((P/'study_plan.json').read_text());fixture=np.fromfile(P/'input_u32.bin',dtype='<u4');oracle=np.fromfile(P/'oracle_u8.bin',dtype='u1');prefix=[]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for c in plan['cases']:
 data=np.zeros(c['allocated_input_words'],dtype='<u4');data[:c['fixture_input_words']]=fixture[c['input_offset']:c['input_offset']+c['fixture_input_words']];prefix.extend(data);prefix.extend(fixture[c['table_offset']:c['table_offset']+1024]);prefix.extend([c[k]for k in ['id','lambda_bits','output_bytes','compressed_words','num_subseq','num_blocks','first_state','first_bit']]+[1024,11,4,128,c['sync_slots'],c['allocated_input_words'],0,0])
prefix=np.array(prefix,dtype='<u4');assert len(prefix)==plan['input_prefix_words'];captures=[];warm=[];checkedbytes=0

def readstate(file,c):
 with file.open('rb')as f:
  header=np.fromfile(f,dtype='<i4',count=7);assert len(header)==7 and header[2]==c['id'] and np.array_equal(header[3:],[c['num_subseq'],c['sync_slots'],c['num_blocks'],c['output_bytes']]);sync=np.fromfile(f,dtype='<u4',count=c['sync_slots']*4).reshape(-1,4);aux=np.fromfile(f,dtype='<u4',count=c['sync_slots']);flags=np.fromfile(f,dtype='u1',count=c['num_blocks']);out=np.fromfile(f,dtype='u1');assert len(out)==(c['output_bytes']if header[0]==4 else 0)
 n=c['num_subseq'];assert np.all(sync[:n,0]<4) and np.all(sync[:n,1]<32) and np.all((sync[:n,3]>=1024)&(sync[:n,3]<2048)) and np.isin(flags,[0,1]).all()
 return header,sync,aux,flags,out
for c in plan['cases']:
 pre=None;prefix_state=None;steps=sorted(DIAG.glob(f"warm_case{c['id']}_step*.bin"),key=lambda p:int(p.stem.split('step')[1]));ops=[]
 for f in steps:
  hd,sy,aux,flags,out=readstate(f,c);op=int(hd[0]);ops.append(op)
  if op==2:pre=aux[:c['num_subseq']].copy();assert np.array_equal(pre,sy[:c['num_subseq'],2]);prefix_state=sy.copy()
  if op==3:
   expected=np.r_[np.uint32(0),np.cumsum(pre[:-1],dtype='u8')].astype('<u4');assert np.array_equal(aux[:c['num_subseq']],expected)and np.array_equal(sy[:c['num_subseq'],2],expected);assert np.array_equal(sy[:,[0,1,3]],prefix_state[:,[0,1,3]])
  if op==4:assert np.array_equal(out,oracle[c['oracle_offset']:c['oracle_offset']+c['output_bytes']]);checkedbytes+=len(out)
  warm.append(dict(case=c['id'],op=op,path=str(f),sha256=sha(f),bytes=f.stat().st_size))
 assert ops[0]==0 and ops[-3:]==[2,3,4] and set(ops)=={0,1,2,3,4}
 for trial in range(1,6):
  f=DIAG/f"trial{trial}_case{c['id']}_final.bin";hd,sy,aux,flags,out=readstate(f,c);n=c['num_subseq'];assert np.array_equal(sy[:n,2],aux[:n]);assert aux[0]==0 and np.all(np.diff(aux[:n].astype('i8'))>=0)and np.all(aux[:n]<=c['output_bytes']);assert np.all(flags[1:c['num_blocks']-1]==1);assert np.array_equal(out,oracle[c['oracle_offset']:c['oracle_offset']+c['output_bytes']]);checkedbytes+=len(out);captures.append(dict(case=c['id'],trial=trial,path=str(f),sha256=sha(f),bytes=f.stat().st_size,convergence_iterations=int(hd[1]),last_block_flag=int(flags[-1]),logical_subsequence_states=n,padding_slots_retained=c['sync_slots']-n))
primary=[]
for trial in range(1,6):
 f=D/f'trial{trial}.bin';raw=np.fromfile(f,dtype='<u4');assert np.array_equal(raw[:len(prefix)],prefix);p=len(prefix)
 for c in plan['cases']:
  n=c['output_bytes']//4;out=raw[p:p+n].view('u1');p+=n;assert np.array_equal(out,oracle[c['oracle_offset']:c['oracle_offset']+c['output_bytes']]);checkedbytes+=len(out)
 # Per-case guards are outputbytes165, input/table/sync/aux wordSENT, flagsbytes165.
 expected=np.tile(np.r_[np.full(16,165,dtype='<u4'),np.full(64,0x4e5a5a5a,dtype='<u4'),np.full(16,165,dtype='<u4')],15);assert np.array_equal(raw[p:],expected);primary.append(dict(trial=trial,count=len(raw),sha256=sha(f),full_decoded_bytes=len(oracle),exact_guard_values=len(expected)))
report=dict(correct=True,full_primary_artifacts=primary,all_actual_final_scratch_captures=captures,all_actual_untimed_stage_captures=warm,decoded_bytes_independently_checked=checkedbytes,independent_oracle='generator independentserialbitreader/tabletransducer checks everybyteagainst reversedoriginalsourceMT19937data; thisauditcomparesallactualGPUbytes, notentropy/sizes',scratch_contract='allrawscratchretained/statebounds/prefix/convergencechecked; scheduling-dependent diagnostic scratchnotprimarynumericaloutput, no canonicalization',original_last_block_convergence_flag_unchecked_policy_preserved=True)
(D/'independent_cpu_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('ANS_CPU_AUDIT',len(captures),len(warm),checkedbytes,flush=True)
