#!/usr/bin/env python3
"""CPU-only full-buffer scientific audit of the fixed supplied educational fixture."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];MASK=np.uint64(0x7fffffff)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def fixtures():
 s=(R/'src/columnarSolver-cuda/main.cu').read_text();cipher=np.array([ord(x)-97 for x in re.search(r'#define ENCRYPTED_T "([^"]+)"',s)[1]],dtype=np.int32);plain=np.array([ord(x)-97 for x in re.search(r'#define DECRYPTED_T "([^"]+)"',s)[1]],dtype=np.int32);scores=np.empty(676,np.float32);seen=set()
 for line in (R/'src/columnarSolver-cuda/data').read_text().splitlines():
  k,v=line.split();assert len(k)==2 and k not in seen;seen.add(k);scores[(ord(k[0])-97)*26+ord(k[1])-97]=v
 assert len(seen)==676 and np.isfinite(scores).all()and np.all(scores>0);return cipher,plain,scores
def parts(raw):return raw[:1024],raw[1024:407552].reshape(1024,397).view(np.int32),raw[407552:].reshape(1024,63)
def sums(out,scores):
 terms=scores[out[:,:-1]*26+out[:,1:]];f=np.zeros(1024,np.float32)
 for j in range(396):f=f+terms[:,j]
 return f,terms.sum(axis=1,dtype=np.float64)
def initial_keys(seeds):
 state=seeds.copy();key=np.broadcast_to(np.arange(30,dtype=np.int32),(1024,30)).copy();rows=np.arange(1024)
 for j in range(30):
  state=(state*np.uint32(26757677)+np.uint32(1))&np.uint32(0x7fffffff);rnd=state.astype(np.float32)/np.float32(2147483648);index=np.minimum((rnd*np.float32(30)).astype(np.int32),29);a=key[:,j].copy();b=key[rows,index].copy();key[:,j]=b;key[rows,index]=a
 return key
def jumped(seeds,draws):
 n=draws.copy().astype(np.uint64);mul=np.uint64(26757677);add=np.uint64(1);am=np.ones(1024,np.uint64);aa=np.zeros(1024,np.uint64)
 while np.any(n):
  flag=(n&np.uint64(1))!=0;aa[flag]=(mul*aa[flag]+add)&MASK;am[flag]=(mul*am[flag])&MASK;add=(mul*add+add)&MASK;mul=(mul*mul)&MASK;n>>=np.uint64(1)
 return ((am*seeds.astype(np.uint64)+aa)&MASK).astype(np.uint32)
def math_gate(raw,cipher,plain,scores,strict=True):
 seeds,out,diag=parts(raw);assert out.min()>=0 and out.max()<26;key=diag[:,30:60].astype(np.int32);perm=np.all(np.sort(key,axis=1)==np.arange(30));assert perm
 assert np.array_equal(diag[:,:30],initial_keys(seeds))
 order=np.argsort(key,axis=1);lengths=13+(np.arange(30)<7);offsets=np.empty((1024,30),np.int32);offsets[np.arange(1024)[:,None],order]=np.cumsum(lengths[order],axis=1)-lengths[order]
 indexes=offsets[:,np.arange(397)%30]+(np.arange(397)//30)[None,:];expected=cipher[indexes];bad_rows=int(np.count_nonzero(np.any(expected!=out,axis=1)));bad_values=int(np.count_nonzero(expected!=out))
 fs,ds=sums(out,scores);actual=diag[:,60].copy().view(np.float32);score_bad=int(np.count_nonzero(fs!=actual));err=np.abs(actual.astype(float)-ds);gamma=(396*2.0**-24)/(1-396*2.0**-24);gamma_ok=bool(np.all(err<=gamma*ds));source_checker_winner=int(np.argmin(fs));source_checker_pass=bool(np.array_equal(out[source_checker_winner],plain));diagwinner=int(np.argmin(actual));known=bool(np.array_equal(out[diagwinner],plain));draws=diag[:,62];assert draws.min()>=600030 and draws.max()<=2550030;assert np.array_equal(jumped(seeds,draws),diag[:,61]);assert source_checker_pass and known
 result={'finalkey_permutations':bool(perm),'initialkey_independent_LCG_shuffle_exact':True,'key_to_plaintext_bad_rows':bad_rows,'key_to_plaintext_bad_values':bad_values,'Float32_score_inconsistent_rows':score_bad,'max_abs_F32_vs_F64_score_error':float(err.max()),'positive_Float32_sum_gamma_bound':gamma_ok,'known_plaintext_diagwinner':diagwinner,'source_hostchecker_replay_winner':source_checker_winner,'source_hostchecker_replay_pass':source_checker_pass,'independent_LCG_skip_ahead_final_state_exact':True,'draw_range':[int(draws.min()),int(draws.max())],'all_full_math_checks_pass':bad_rows==0 and score_bad==0 and gamma_ok and known}
 if strict:assert result['all_full_math_checks_pass'],result
 return result
def main():
 a=argparse.ArgumentParser();a.add_argument('--result',type=Path,required=True);a.add_argument('--output',type=Path,required=True);z=a.parse_args();rp=z.result.resolve();r=json.load(open(rp));c=json.load(open(rp.with_name('config.json')));assert r['matched_performance_valid']and c['native_reference_repaired'];cipher,plain,scores=fixtures();goldp=Path('/tmp/columnarSolver_native/cpu_separate.bin');gold=np.fromfile(goldp,'<u4');setup=np.arange(1024,dtype=np.uint32)
 for i in range(1023):setup[i+1:]=(setup[i+1:]*np.uint32(26757677)+np.uint32(1))&np.uint32(0x7fffffff)
 assert np.array_equal(setup,gold[:1024]);goldmath=math_gate(gold,cipher,plain,scores);reviews=[];bindings=[];digests=set()
 for model,v in r['models'].items():
  assert len(v['trials'])==5 and v['status']=='native_check_pass'
  for path,expected in v['adapter_file_hashes'].items():
   h=sha(path);assert h==expected,(model,path);bindings.append({'model':model,'path':path,'sha256':h})
  for t in v['trials']:
   p=Path(t['raw_state_path']);raw=np.fromfile(p,'<u4');assert np.array_equal(raw,gold);assert t['source_CPU_exact'];review=math_gate(raw,cipher,plain,scores);g=np.fromfile(t['guard_path'],'u1');assert len(g)==2560 and np.all(g==0xa7);artifact=t['output_artifact'];assert sha(artifact['saved_path'])==artifact['sha256'];actual=np.fromfile(artifact['saved_path'],'<f8');seeds,out,diag=parts(raw);expected=np.concatenate([scores.astype(float),cipher.astype(float),seeds.astype(float),out.ravel().astype(float),diag.ravel().astype(float),g.astype(float)]);assert len(actual)==475697 and np.array_equal(actual,expected);assert np.array_equal(actual[:2097],np.concatenate([scores,cipher,seeds]).astype(float));h=sha(p);digests.add(h);reviews.append({'model':model,'trial':t['trial'],'time_seconds':t['time_seconds'],'raw_state_path':str(p),'raw_state_sha256':h,'raw_bytes':p.stat().st_size,'artifact_sha256':artifact['sha256'],'full_encoded_values':475697,'exact_actual_input_prefix_values':2097,'physical_guard_bytes_checked':2560,'math':review});print(model,t['trial'],'fullPASS',flush=True)
 controls=[]
 for name in ['diagnostic01','diagnostic_nvcc129','diagnostic_storage_v2']:
  p=Path('/tmp/columnarSolver_native')/name/'trial0_state.bin';raw=np.fromfile(p,'<u4');review=math_gate(raw,cipher,plain,scores,strict=False);controls.append({'name':name,'raw_path':str(p),'sha256':sha(p),'math':review,'interpretation':'Replayedsourcehostcheckeroncapturedinstrumentedreferenceoutputs, NOT originalsourceCLI/GPUexecution. Nativeoriginalparser/indexboundsnotrununsafely.'})
 assert controls[0]['math']['key_to_plaintext_bad_rows']==489 and controls[1]['math']['key_to_plaintext_bad_rows']==489 and controls[2]['math']['all_full_math_checks_pass']
 result={'status':'all20_exact_full_source_workload_independent_saved_audit_pass','result':str(rp),'result_sha256':sha(rp),'configuration_id':r['configuration_id'],'native_reference_repaired':True,'exclude_untouched_native_aggregates':True,'full_workload':{'KEY_LENGTH':30,'CLIMBINGS':150000,'trajectories':1024,'plaintext_length':397},'all20_rawstates_sha256_distinct_count':len(digests),'full_values_compared':20*475697,'actual_physical_guard_bytes_checked':20*2560,'source_execution_bindings':bindings,'binding_count':len(bindings),'trial_reviews':reviews,'CPU_source_gold':{'path':str(goldp),'sha256':sha(goldp),'math':goldmath},'native_intermediate_storage_controls':controls,'scientific_support_boundary':'Exactfixedsuppliedacademicfixture and independentlyvalidatedpermutation/objective/knownplaintextcertificate; notglobaloptimality/otherciphertext/keylength/externalworkload/cryptographicRNGquality. Sourceboundsparserinterventionsandprivate-storagecodegencondition explicit.','raw_artifacts_unchanged':True};z.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
