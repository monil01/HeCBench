#!/usr/bin/env python3
"""CPU-only full replay of immutable SS trial artifacts and physical state."""
from pathlib import Path
import argparse,json,hashlib,array,statistics
P=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def words(b):
 a=array.array('I');a.frombytes(b);assert a.itemsize==4;return a

def main():
 a=argparse.ArgumentParser();a.add_argument('result',type=Path);a.add_argument('--fixtures',type=Path,default=Path('/tmp/hecbench-stencil-ports-ss-fixtures'));a.add_argument('--output',type=Path,default=P/'all4_full_capture_audit.json');z=a.parse_args();r=json.load(open(z.result));c=json.load(open(z.result.resolve().with_name('config.json')));F=z.fixtures;cases=json.load(open(F/'cases.json'));assert len(cases)==7 and r['matched_performance_valid']and r['reference_agreement']and all(r['variant_verified'].values());gold=[]
 for case in cases:
  text=Path(case['text_path']).read_bytes();pattern=case['pattern_ascii'].encode()
  if pattern is None:pattern=next(line.split('\t')[3].encode()for line in(F/'cases.tsv').read_text().splitlines()if line.split('\t')[2]==case['text_path'])
  lower=bytes(x+32 if 65<=x<=90 else x for x in text);pat=bytes(x+32 if 65<=x<=90 else x for x in pattern);g=array.array('I',(i for i in range(len(text)-len(pat)+1)if lower[i:i+len(pat)]==pat));assert g==words(Path(case['oracle_path']).read_bytes());gold.append((text,pattern,g))
 reviews=[];physical_count=0
 for m,v in r['models'].items():
  assert v['complete']and v['provenance_valid']and v['status']=='native_check_pass'and len(v['trials'])==5
  assert v['mean_seconds']==statistics.mean(t['time_seconds']for t in v['trials'][1:])
  for t in v['trials']:
   assert t['correct']and t['jit_excluded']and t['timing_region']=='full_source_repeat_loops_host_launch_control_and_context_completion'
   prefix=array.array('I');output=array.array('I')
   for row,(text,pattern,g)in zip(t['cases'],gold):
    idx=row['case'];n=len(text);p=len(pattern);N=n-p+1;groups=(N+131071)//131072;assert row['matches']==len(g)and row['repeats']==cases[idx]['repeats'];captures={x['role']:x for x in row['complete_physical_captures']};data={}
    for role,d in captures.items():
     path=Path(d['path']);assert path.stat().st_size==d['bytes']and sha(path)==d['sha256'];data[role]=path.read_bytes();physical_count+=1
    before=[data['before_allocation'+str(k)]for k in range(4)];after=[data['after_allocation'+str(k)]for k in range(4)];assert before[0][256:-256]==text and before[1][256:-256]==pattern
    for bb in before+after:assert bb[:256]==bytes([0xa5])*256 and bb[-256:]==bytes([0x5a])*256
    assert before[0]==after[0]and before[1]==after[1]and all(bb[256:-256]==bytes([255])*(len(bb)-512)for bb in before[2:])
    prefix.extend([idx,n,p,cases[idx]['repeats'],groups,N,131072,256,0xffffffff]);prefix.extend(before[0]);prefix.extend(before[1]);prefix.extend(words(before[2]));prefix.extend(words(before[3]))
    counts=words(after[2][256:-256]);raw=words(after[3][256:-256]);assert len(counts)==groups and len(raw)==N
    expected=[0]*groups
    for pos in g:expected[pos//131072]+=1
    assert list(counts)==expected;pairs=[]
    for group,count in enumerate(counts):
     start=group*131072;capacity=min(131072,N-start)
     for j in range(count):
      pos=raw[start+j];assert start<=pos<start+capacity;pairs.append((pos,start+j))
     assert all(x==0xffffffff for x in raw[start+count:start+capacity])
    pairs.sort();assert [x[0]for x in pairs]==list(g)and words(data['complete_canonical_to_physical_slot_permutation_u32'])==array.array('I',(x[1]for x in pairs))
    reconstructed=array.array('I',[0xffffffff])*N
    for pos,slot in pairs:reconstructed[slot]=pos
    assert reconstructed==raw
    host=array.array('I',raw);written=counts[0]
    for group,count in enumerate(counts[1:],1):host[written:written+count]=raw[group*131072:group*131072+count];written+=count
    host[:written]=array.array('I',sorted(host[:written]));assert host==words(data['complete_source_host_final_result_u32'])and host[:written]==g
    output.extend([idx,groups,len(g)]);output.extend(counts);output.extend(g)
   assert len(prefix)==c['input_prefix_count']and prefix==words((F/'canonical_actual_input_prefix.u32.bin').read_bytes())
   descriptor=t['output_artifact'];path=Path(descriptor['saved_path']);assert sha(path)==descriptor['sha256'];expected=prefix+output;assert path.stat().st_size==len(expected)*4 and path.read_bytes()==expected.tobytes();assert len(expected)==t['checked_values']==5533660
   reviews.append(dict(model=m,trial=t['trial'],artifact=str(path),sha256=sha(path),checked_words=len(expected),input_words=len(prefix),all7cases=True,all_physical_bytes_hashed=True,source_final_exact=True,complete_raw_reconstruction=True,complete_host_tail_replay=True,guards_inactive_inputs=True))
 out=dict(benchmark='ss',pass_all=True,artifact_count=len(reviews),physical_capture_files_hashed=physical_count,result_sha256=sha(z.result),oracle='Independentevery-startASCIIbytecomparison(fullpositions), notGPUfilters/BMH',raw_output_identity_claim=False,scientific_scope='Textmatchingnotphysicalmodel; exactsourcefinalsortedpositionswithcompletephysicalbijection, conditionalinitializedoutputandhostmemmove.',trials=reviews);z.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='trials'}))
if __name__=='__main__':main()
