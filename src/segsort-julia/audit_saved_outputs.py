"""CPU-only audit of immutable saved artifacts; never launches or retimes GPU work."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
P=Path(__file__).resolve().parent
R=P.parent.parent
F=Path(sys.argv[1]).resolve()
r=json.loads(F.read_text());meta=json.loads((P/'fixture_provenance.json').read_text());n=meta['N'];ns=meta['segments'];counts=np.array(meta['all13bins_counts'],dtype='i4');off=np.fromfile(P/'offsets_i32.bin',dtype='<i4');lens=np.diff(off)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[np.fromfile(P/f'keys_case{i}_i32.bin',dtype='<i4')for i in [1,2]];vals=np.fromfile(P/'values_case2_u64.bin',dtype='<u8');prefix=[]
for mode in [0,1]:
 prefix.append(inputs[mode].view('<u4'))
 if mode:prefix.append(vals.view('<u4'))
 prefix.append(off.view('<u4'));prefix.append(np.array([n,ns,mode,13,2048,5000,1,0],dtype='<u4'))
prefix=np.concatenate(prefix);assert len(prefix)==1430962
expected=[]
for mode in [0,1]:
 expected.append(np.fromfile(P/f'oracle_case{mode+1}_i32.bin',dtype='<i4').view('<u4'))
 if mode:expected.append(np.fromfile(P/'oracle_values_case2_u64.bin',dtype='<u8').view('<u4'))
 totals=np.r_[0,np.cumsum(counts)];counter=[]
 for i in range(14):counter.extend([int(totals[i+1])if i<13 else 5000,int(totals[i])if i<13 else ns])
 expected.append(np.array(counter,dtype='<u4'))
expected=np.concatenate(expected);assert len(expected)==1430248
bindings={};rows=[];firsthash=None
for model,v in r['models'].items():
 if v['status']=='missing':continue
 assert v['status']=='native_check_pass'and len(v['trials'])==5
 for path,h in v['adapter_file_hashes'].items():assert sha(path)==h;bindings[path]=h
 for t in v['trials']:
  a=t['output_artifact'];assert sha(a['saved_path'])==a['sha256'];raw=np.fromfile(a['saved_path'],dtype='<u4');assert len(raw)==2861466 and raw[:len(prefix)].tobytes()==prefix.tobytes();actual=raw[len(prefix):len(prefix)+len(expected)];assert np.array_equal(actual,expected)
  if firsthash is None:firsthash=a['sha256']
  assert firsthash==a['sha256']
  pos=0;assoc=0;segments=0
  for mode in [0,1]:
   keys=actual[pos:pos+n].view('<i4');pos+=n
   if mode:payload=actual[pos:pos+2*n].view('<u8');pos+=2*n
   for s,e in zip(off[:-1],off[1:]):
    assert np.all(keys[s+1:e]>=keys[s:e-1])if e>s else True
    assert np.array_equal(np.sort(keys[s:e]),np.sort(inputs[mode][s:e]))
    if mode:
     idx=(payload[s:e]-np.uint64(0xfedc000000000000)).astype('i8');assert np.all((idx>=s)&(idx<e));assert np.array_equal(np.sort(idx),np.arange(s,e));assert np.array_equal(inputs[mode][idx],keys[s:e]);assoc+=int(e-s)
    segments+=1
   pos+=28
  assert pos==len(expected)
  g=raw[-256:];single=np.full(6*16,0x5a5a5a5a,dtype='<u4');pair=np.r_[single,np.tile(np.array([0xfedcba98,0x12345678],dtype='<u4'),32)];assert np.array_equal(g,np.r_[single,pair])
  rawids=t['internal_bin_ids_raw_not_public_output'];assert len(rawids)==2
  for ids in rawids:
   assert sorted(ids)==list(range(ns));start=0
   for b,c in enumerate(counts):
    for sid in ids[start:start+c]:assert min(0 if lens[sid]<=1 else int(lens[sid]-1).bit_length(),12)==b
    start+=int(c)
  rows.append(dict(model=model,trial=t['trial'],artifact_sha256=a['sha256'],artifact_count=len(raw),exact_input_bytes=4*len(prefix),full_output_and_counter_words=len(expected),guards=256,segments_independently_checked=segments,key_payload_associations_checked=assoc,raw_internal_bin_ids_checked=2*ns,sortedness=True,multiset_preserved=True,key_value_association=True,all_guard_words_exact=True))
proof=dict(result_path=str(F.relative_to(R)),result_sha256=sha(F),NumPy=np.__version__,method='Rehash every full artifact and declared executed binding; exact source input/controls prefix and CPU independently per-segment sorted outputs; alternate sortedness/multiset checks plus derive UInt64 payload index and prove segment bounds/permutation/original key association. Internal atomic bin-ID order checked without canonicalizing public outputs.',rows=rows,executed_source_bindings=bindings,identical_artifact_sha256=firsthash,limits=['Unique pair fixture keys exclude unspecified duplicate-key payload tie ordering.','Independent CPU sort and permutation algebra do not prove all unmeasured input distributions/sentinel keys.'])
(P/'saved_output_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print('AUDITED',len(rows),'full artifacts;',len(bindings),'bindings;',len(prefix),'prefix words;',len(expected),'output/counter words; payload associations',n,'each')
