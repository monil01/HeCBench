#!/usr/bin/env python3
"""Full saved-value audit, independent coordinate DP, and CPU cold regeneration."""
import pathlib,json,array,hashlib,importlib.util,subprocess,tempfile,shutil,sys
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent;S=R/'.porting-state/portable_julia';F=pathlib.Path(sys.argv[1]);F=F if F.is_absolute()else R/F;r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());N=plan['cases'];NA,NB=plan['sequence_bytes'];PREFIX=plan['input_prefix_count'];COUNT=plan['artifact_count'];CI=NA+NB+989
spec=importlib.util.spec_from_file_location('oracle',P/'native_diagnostics/audit_cli.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def sha(b):return hashlib.sha256(b).hexdigest()
def readarr(b):a=array.array('i');a.frombytes(b);return a
expected=readarr((P/'oracle_i32.bin').read_bytes());swappedexpected=readarr((P/'native_diagnostics/orientation_swapped_expected_i32.bin').read_bytes());oracle_cache={}
def mathematical_check(a,orientation):
 key=sha(a.tobytes())
 if key in oracle_cache:return oracle_cache[key]
 assert len(a)==COUNT
 maxa,maxb=(160,128)if orientation=='primary'else(128,160);na,nb=(NA,NB)if orientation=='primary'else(NB,NA)
 assert list(a[:7])==[N,maxa,maxb,-6,-1,128,128]
 ptr=7+CI;parts=[]
 for length,guard in [(na,211),(nb,211),(N,24681357),(N,24681357),(576,-12345),(91,-12345)]:
  vals=a[ptr:ptr+length+32];assert list(vals[:16])==list(vals[-16:])==[guard]*16;parts.append(vals[16:-16]);ptr+=length+32
 aa,bb,oa,ob,mat,enc=parts;assert list(mat)==o.mat and list(enc)==o.enc
 outputs=[[]for _ in range(5)];prevA=prevB=0;all_ties=[]
 for k in range(N):
  x=bytes(list(aa[prevA:oa[k]])).decode();y=bytes(list(bb[prevB:ob[k]])).decode();prevA=oa[k];prevB=ob[k];score,(ae,be),ties=o.sw(x,y);back,(ac,bc),rt=o.sw(x[:ae][::-1],y[:be][::-1],True);assert score==back
  for dst,v in zip(outputs,[ae-ac,ae,be-bc,be,score]):dst.append(v)
  all_ties.append([ties,rt])
 assert list(a[ptr:ptr+N])==outputs[1]and list(a[ptr+N:ptr+2*N])==outputs[3];ptr+=2*N
 assert list(a[ptr:PREFIX])==[-12345]*(5*(N+32));ptr=PREFIX
 for dst in outputs:
  vals=a[ptr:ptr+N+32];assert list(vals)==[-12345]*16+dst+[-12345]*16;ptr+=N+32
 assert ptr==COUNT
 out=dict(orientation=orientation,independent_DP_pairs=N,all5_fields_exact=True,maximum_score=max(outputs[4]),forward_reverse_tied_maxima=all_ties);oracle_cache[key]=out;return out
checked=[];bindings={};hashes=set();prefixes=set()
assert r['reference_agreement']and r['matched_performance_valid']
for model,v in r['models'].items():
 assert v['complete']and v['timing_valid']and v['provenance_valid']and len(v['trials'])==5
 for file,hash in v['adapter_file_hashes'].items():
  assert sha(pathlib.Path(file).read_bytes())==hash,file;bindings[file]=hash
 for t in v['trials']:
  f=pathlib.Path(t['output_artifact']['saved_path']);b=f.read_bytes();a=readarr(b);assert sha(b)==t['output_artifact']['sha256']and a==expected;maths=mathematical_check(a,'primary');assert t['correct']and t['jit_excluded']and t['time_seconds']>0
  hashes.add(sha(b));prefixes.add(sha(b[:PREFIX*4]));checked.append(dict(model=model,trial=t['trial'],artifact=str(f.relative_to(R)),sha256=sha(b),count=len(a),full_actual_input_prefix_count=PREFIX,all_fields_exact=True,independent_math_result=maths['orientation']))
assert len(checked)==20 and len(hashes)==len(prefixes)==1
swapped=[]
for model in ['cuda','cuda_jl','ka','jacc']:
 f=R/f'.porting-state/diagnostics/bsw/swapped_{model}_i32.bin';b=f.read_bytes();a=readarr(b);assert a==swappedexpected;maths=mathematical_check(a,'swapped');log=R/f'.porting-state/diagnostics/bsw/swapped_{model}.log';assert 'correct=true' in log.read_text();swapped.append(dict(model=model,diagnostic_only=True,timing_excluded=True,pipeline_executions=1,forward_reverse_kernel_launches=2,artifact=str(f.relative_to(R)),sha256=sha(b),checked_values=len(a),log=dict(path=str(log.relative_to(R)),sha256=sha(log.read_bytes())),all5_fields_exact=True))
assert len({v['sha256']for v in swapped})==1
names=['reference.fasta','query.fasta','fixture.json','oracle_i32.bin','study_params.h','study_params.jl','study_plan.json','native_diagnostics/orientation_swapped_expected_i32.bin','native_diagnostics/orientation_swapped_params.jl','native_diagnostics/orientation_swapped.cu','native_diagnostics/orientation_swapped_fixture.json']
with tempfile.TemporaryDirectory(prefix='bsw-cold-cpu-')as tmp:
 q=pathlib.Path(tmp);t=q/'src/bsw-julia';t.mkdir(parents=True);shutil.copytree(R/'src/bsw-cuda',q/'src/bsw-cuda');shutil.copytree(P/'native_diagnostics',t/'native_diagnostics');shutil.copy(P/'generate_assets.py',t/'generate_assets.py');shutil.copy(P/'study_native.cu',t/'study_native.cu')
 subprocess.run([sys.executable,str(t/'generate_assets.py')],cwd=q,check=True,stdout=subprocess.DEVNULL);subprocess.run([sys.executable,str(t/'native_diagnostics/prepare_swapped_orientation.py')],cwd=q,check=True)
 cold=[]
 for name in names:
  a=(P/name).read_bytes();b=(t/name).read_bytes();assert a==b,name;cold.append(dict(path=name,bytes=len(a),sha256=sha(a),byte_exact=True))
out=dict(benchmark='bsw',result=str(F.relative_to(R)),result_sha256=sha(F.read_bytes()),status='pass',primary_GPU_executions_added=0,supplemental_untimed_pipeline_processes_per_model=1,primary_all20_artifacts_bit_identical=True,unique_primary_artifact_sha256=sorted(hashes),primary_artifact_values=COUNT,actual_input_prefix_values=PREFIX,source_bindings=bindings,trials=checked,independent_math=dict(primary=next(v for v in oracle_cache.values()if v['orientation']=='primary'),swapped=next(v for v in oracle_cache.values()if v['orientation']=='swapped'),cache_note='Eachfilefullyread/rehash/comparedtofullindependentoracle; mathematicalDP reused onlyforidenticalwholeartifactbytes.65distinctpairDPchecksperorientation;no claimof20distinctinputsets.'),supplemental_swapped_orientation=swapped,cold_reproduction=cold,precision_bound='LargestsourceInt16score<=11*128=1408; ASCIIuppercaseinputswithin91entryencodingtable. Originalzero-score/overflow/longqueryhostallocation/fullDVCdatasetsnotcertified.',timing_qualification='Allprimarytimessavedunchanged; oldnative-onlyprecomputedoffsetsseriesexplicitlyexcluded; no fastestselection ordiagnostictimepromotion.')
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status='pass',primary_captures=len(checked),source_bindings=len(bindings),swapped_diagnostics=len(swapped),cold_exact_files=len(cold))))
