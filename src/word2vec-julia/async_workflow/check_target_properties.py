#!/usr/bin/env python3
"""Full saved asynchronous workflow properties; never raw-model equivalence."""
import argparse,json,pathlib,hashlib,ctypes
import numpy as np
P=pathlib.Path(__file__).resolve().parent
D=128;N=1025;WORDS=8200

def check(folder):
 folder=pathlib.Path(folder); failures=[];checks={}
 def require(name,condition,**metrics):
  checks[name]={'pass':bool(condition),**metrics}
  if not condition:failures.append(name)
 def array(name,dtype,count):
  a=np.fromfile(folder/name,dtype=dtype);require(name+'_size',len(a)==count,actual=len(a),expected=count);return a
 vocab=[line.split('\t') for line in (folder/'vocab.tsv').read_text().splitlines()];names=[x[0] for x in vocab];counts=np.array([int(x[1]) for x in vocab]);require('vocabulary',len(names)==N and names[0]=='</s>' and set(names[1:])=={f'w{i:04}'for i in range(1,1025)} and counts[0]==1 and np.all(counts[1:]==512))
 initial=array('initial_syn0.f32','<f4',N*D).reshape(N,D);zero=array('initial_syn1.f32','<f4',N*D);syn0=array('final_syn0.f32','<f4',N*D).reshape(N,D);syn1=array('final_syn1.f32','<f4',N*D).reshape(N,D)
 rng=1;gold=np.empty(N*D,dtype=np.float32)
 for i in range(len(gold)):
  rng=(1664525*rng+1013904223)&0xffffffff;gold[i]=((np.float32(rng&65535)/np.float32(65536))-np.float32(.5))/np.float32(D)
 require('initial_syn0_exact_LCG',np.array_equal(initial.ravel(),gold));require('initial_syn1_zero',np.all(zero==0));require('initial_sentence_defined',np.all(array('initial_sen.i32','<i4',6150)==0))
 require('full_model_finite',np.isfinite(syn0).all() and np.isfinite(syn1).all());require('model_changed',not np.array_equal(syn0,initial) and np.any(syn1!=0))
 table=np.memmap('/tmp/w2vfull/table.i32',dtype='<i4',mode='r');require('table_count',len(table)==100000000);require('table_valid_monotonic',np.min(table)>=0 and np.max(table)<N and np.all(table[1:]>=table[:-1]));hist=np.bincount(table,minlength=N);prob=counts.astype(float)**.75;prob/=prob.sum();require('table_histogram',np.max(np.abs(hist-prob*len(table)))<=3,max_count_error=float(np.max(np.abs(hist-prob*len(table)))))
 proof=json.load(open(folder/'table_full_byte_comparison.json'));require('actual_table_full_byte_comparison',proof['equal'] and proof['actual_device_bytes_compared']==400000000)
 libc=ctypes.CDLL(None);libc.srand(123);expected_rng=np.array([libc.rand() for _ in range(1024)],dtype=np.uint32);actual_rng=array('initial_rng.u32','<u4',1024);require('initial_rng_libc',np.array_equal(actual_rng,expected_rng))
 actual_exp=array('exptable.f32','<f4',1000);arg=(np.arange(1000,dtype=np.float32)/np.float32(1000)*np.float32(2)-np.float32(1))*np.float32(6);e=np.exp(arg.astype(float)).astype(np.float32);oracle_exp=(e/(e+np.float32(1))).astype(np.float32);require('constant_sigmoid_table_full',np.max(np.abs(actual_exp-oracle_exp))<=1.2e-7,max_abs_error=float(np.max(np.abs(actual_exp-oracle_exp))))
 # Deterministic fresh single-position control against independent Float64 SGD.
 us=array('unit_sen.i32','<i4',6150);u0=initial.astype(float).copy();u1=np.zeros_like(u0);state=int(actual_rng[0]);state=(1664525*state+1013904223)&0xffffffff;b=state%8;positions=[a-8 for a in range(b,17-b)if a!=8 and 0<=a-8<1024];rows=us[np.array(positions)];neu=u0[rows].mean(axis=0);err=np.zeros(D,dtype=float);minimum_lookup_margin=1.
 for d in range(26):
  if d==0:target=int(us[0]);label=1
  else:
   state=(1664525*state+1013904223)&0xffffffff;target=int(table[state%100000000]);target=state%1024+1 if target==0 else target
   if target==us[0]:continue
   label=0
  f=float(np.dot(neu,u1[target]));lookup=(f+6)*83;minimum_lookup_margin=min(minimum_lookup_margin,abs(lookup-round(lookup)))
  sig=1. if f>6 else 0. if f< -6 else float(actual_exp[int(lookup)]);g=(label-sig)*.05;err+=g*u1[target];u1[target]+=g*neu
 for row in rows:u0[row]+=err
 unit0=array('unit_syn0.f32','<f4',N*D).reshape(N,D);unit1=array('unit_syn1.f32','<f4',N*D).reshape(N,D);ratio=max(float(np.max(np.abs(unit0-u0)/(1e-6+2e-5*np.abs(u0)))),float(np.max(np.abs(unit1-u1)/(1e-6+2e-5*np.abs(u1)))));require('fresh_unit_gradient_full_models',ratio<=1,max_budget_ratio=ratio,max_abs_error=max(float(np.max(np.abs(unit0-u0))),float(np.max(np.abs(unit1-u1)))),lookup_margin=minimum_lookup_margin)
 unitrng=array('unit_rng.u32','<u4',1024);ur=actual_rng.copy();ur[0]=state;require('fresh_unit_rng',np.array_equal(unitrng,ur))
 meta=json.load(open(folder/'trace_meta.json'));calls=meta['callbacks'];require('trace_complete',0<calls<=2048 and meta['execution_count']==calls and meta['trace_words']==WORDS,callbacks=calls)
 trace=array('callbacks.u32','<u4',calls*WORDS).reshape(calls,WORDS);require('execution_order_permutation',np.array_equal(np.sort(trace[:,0]),np.arange(calls,dtype=np.uint32)));trace=trace[np.argsort(trace[:,0])];badids=0;badalpha=0;chainbad=0;sentence_sum=0;zero_calls=0
 for rec in trace:
  n=int(rec[1]);require('sentence_count_range',0<=n<=6);sentence_sum+=n;zero_calls+=n==0
  raw=rec[2:6152];ids=raw[:n*1024].view(np.int32);badids+=int(np.count_nonzero((ids<0)|(ids>=N)));alpha=raw[6144:6144+n].view(np.float32);badalpha+=int(np.count_nonzero(~np.isfinite(alpha)|(alpha<=0)|(alpha>np.float32(.05))))
  before=rec[6152:7176];after=rec[7176:8200];chainbad+=int(np.count_nonzero(before!=expected_rng))
  for _ in range(26*n):expected_rng=expected_rng*np.uint32(1664525)+np.uint32(1013904223)
  chainbad+=int(np.count_nonzero(after!=expected_rng))
 require('actual_consumed_ids_valid',badids==0,bad_values=badids);require('actual_consumed_alpha_range',badalpha==0,bad_values=badalpha);require('all_callback_rng_chain',chainbad==0,mismatches=chainbad,full_sentences=sentence_sum,zero_sentence_calls=zero_calls)
 finalrng=array('final_rng.u32','<u4',1024);require('final_rng_chain',np.array_equal(finalrng,expected_rng))
 target_exp=array('target_exptable.f32','<f4',1000);require('mapped_exptable_exact_readonly',np.array_equal(target_exp,actual_exp))
 guards=array('guards.bin','u1',6*512);require('all_6_actual_allocations_guards_5source_plus_exp_mapping',np.all(guards==0xa7),guard_bytes=len(guards))
 with open(folder/'vectors.bin','rb') as f:
  require('output_header',f.readline()==b'1025 128\n');out=np.empty((N,D),np.float32);file_names=[]
  for i in range(N):
   word=bytearray()
   while True:
    c=f.read(1)
    if c==b' ':break
    if not c:raise ValueError('truncated word')
    word.extend(c)
   file_names.append(word.decode());out[i]=np.frombuffer(f.read(D*4),dtype='<f4');require('output_row_newline_'+str(i),f.read(1)==b'\n')
  require('output_no_extra',f.read()==b'')
 require('full_export_matches_final_GPU',file_names==names and np.array_equal(out,syn0))
 # Full deterministic cyclic-context expected NCE diagnostic; not SGD replay.
 ids=np.array([names.index(f'w{i:04}') for i in range(1,1025)]);context=np.zeros((1024,D),np.float64)
 for shift in list(range(-8,0))+list(range(1,9)):context+=syn0[ids[np.mod(np.arange(1024)+shift,1024)]].astype(float)/16
 scores=context@syn1.astype(float).T
 # Exact expected source table/fallback distribution under uniform UInt32 state.
 edges=np.concatenate([[0],np.cumsum(hist)]);cycles,tail=divmod(2**32,len(table));mass=cycles*hist+np.maximum(0,np.minimum(hist,tail-edges[:-1]));mass[0]=0
 for cycle in range(cycles+1):
  lo=cycle*len(table);hi=min(lo+int(hist[0]),2**32)
  if hi>lo:
   values=np.arange(lo,hi,dtype=np.uint64);mass[1:]+=np.bincount((values%1024).astype(np.int64),minlength=1024)
 negative_prob=mass.astype(float)/(2**32)
 require('independent_sampler_probability',abs(negative_prob.sum()-1)<1e-14)
 # Source skips sampled target==positive; condition term omitted (zero loss).
 penalties=np.logaddexp(0,scores);neg=penalties@negative_prob-negative_prob[ids]*penalties[np.arange(1024),ids]
 losses=(np.logaddexp(0,-scores[np.arange(1024),ids])+25*neg)/26
 # Same skip policy at initial all-zero syn1.
 initial_losses=(1+25*(1-negative_prob[ids]))*np.log(2)/26
 improvement=float(np.mean(initial_losses-losses));np.save(folder/'independent_NCE_loss_rows.npy',np.stack([initial_losses,losses],axis=1))
 require('fixed_expected_loss_improvement',np.isfinite(losses).all() and improvement>=.0001,normalized_mean_initial=float(initial_losses.mean()),normalized_mean_final=float(losses.mean()),improvement=improvement,threshold=.0001)
 result={'pass':not failures,'failures':failures,'checks':checks,'raw_model_identity_claim':False,'numerical_agreement_claim':False,'contract':'bounded_full_asynchronous_workflow_property_only','hashes':{x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in folder.iterdir()if x.is_file()and x.name not in ['properties.json']}}
 (folder/'properties.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('folder',type=pathlib.Path);z=a.parse_args();r=check(z.folder);print(json.dumps({'pass':r['pass'],'failures':r['failures'],'loss':r['checks']['fixed_expected_loss_improvement']}));raise SystemExit(0 if r['pass'] else 1)
