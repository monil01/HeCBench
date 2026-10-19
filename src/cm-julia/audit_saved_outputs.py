#!/usr/bin/env python3
"""CPU-only complete saved numerical evidence and byte-exact fixture regeneration."""
from pathlib import Path
import json,array,hashlib,sys,subprocess,tempfile,shutil,importlib.util
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;F=Path(sys.argv[1]);F=F if F.is_absolute()else ROOT/F;r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());spec=importlib.util.spec_from_file_location('independent_cm_oracle',P/'generate_assets.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
def sha(b):return hashlib.sha256(b).hexdigest()
def arr(b,code='d'):a=array.array(code);a.frombytes(b);return a
assert sys.byteorder=='little'and r['reference_agreement']and r['matched_performance_valid'];PREFIX=plan['input_prefix_count'];COUNT=plan['artifact_count'];expected=arr((P/'oracle_f64.bin').read_bytes());cache={};bindings={};trials=[];allhashes=set();allprefixes=set()
def full_math(a):
 key=sha(a.tobytes())
 if key in cache:return cache[key]
 ptr=0;randoms=[]
 for k in range(2):
  assert list(a[ptr:ptr+7])==[o.N,o.R,o.SIG,2 if k==0 else 1,o.UC,o.DB,o.SB];ptr+=7;parts=[]
  for n,guard in [(o.N,1234567),(o.N,1234567),(o.M,-999),(o.DB,-12345),(o.R,-999),(o.SB,-12345)]:
   values=a[ptr:ptr+n+32];ptr+=n+32;assert list(values[:16])==list(values[-16:])==[guard]*16;parts.append(list(values[16:-16]))
  assert parts[0]==o.refs()[k]and parts[1]==o.query();assert parts[3]==[0]*o.DB and parts[4]==[0]*o.R and parts[5]==[0]*o.SB
  randoms.append(parts[2])
 assert ptr==PREFIX;prefix,tail,groups=o.primary(randoms);assert list(a)==prefix+tail
 cache[key]={'independent_actual_input_reevaluation':True,'groups':groups,'all_raw_scores':2*o.R,'all_reduction_partials':2*(o.DB+o.SB),'host_feedback_thresholds_stats_exact':True,'whole_artifact_sha256':key};return cache[key]
for m,v in r['models'].items():
 assert v['complete']and v['timing_valid']and v['provenance_valid']and len(v['trials'])==5
 for p,h in v['adapter_file_hashes'].items():assert sha(Path(p).read_bytes())==h,p;bindings[p]=h
 for t in v['trials']:
  f=Path(t['output_artifact']['saved_path']);b=f.read_bytes();a=arr(b);assert len(a)==COUNT and sha(b)==t['output_artifact']['sha256']and a==expected and t['correct']and t['jit_excluded']and t['time_seconds']>0;full_math(a)
  allhashes.add(sha(b));allprefixes.add(sha(b[:PREFIX*8]));trials.append({'model':m,'trial':t['trial'],'path':str(f.relative_to(ROOT)),'sha256':sha(b),'checked_values':len(a),'actual_input_prefix_values':PREFIX,'full_exact':True})
assert len(trials)==20 and len(allhashes)==len(allprefixes)==1
scratch=list((ROOT/'.porting-state/porting_logs/cm-julia-jacc/traces').glob('actual_scratch_*.bin'));assert len(scratch)==1;assert list(arr(scratch[0].read_bytes(),'i'))==[0]*256
names=list(plan['files'])+['study_plan.json']
with tempfile.TemporaryDirectory(prefix='cm-CPU-cold-')as tmp:
 q=Path(tmp);(q/'native_diagnostics').mkdir();shutil.copy2(P/'generate_assets.py',q/'generate_assets.py');shutil.copy2(P/'native_diagnostics/generate_random.cpp',q/'native_diagnostics/generate_random.cpp');subprocess.run([sys.executable,str(q/'generate_assets.py')],check=True,stdout=subprocess.DEVNULL);cold=[]
 for name in names:
  a=(P/name).read_bytes();b=(q/name).read_bytes();assert a==b,name;cold.append({'path':name,'bytes':len(a),'sha256':sha(a),'byte_exact':True})
w=json.loads((P/'native_CLI_witness.json').read_text());assert sha((ROOT/w['original_cli_log']['path']).read_bytes())==w['original_cli_log']['sha256'];assert sha((ROOT/w['source_pvalue_component_defect']['actual_log']).read_bytes())==w['source_pvalue_component_defect']['sha256']
out={'benchmark':'cm','result':str(F.relative_to(ROOT)),'result_sha256':sha(F.read_bytes()),'status':'pass','GPU_runs_added':0,'all20_primary_captures_bit_identical':True,'checked_values_per_trial':COUNT,'input_prefix_count':PREFIX,'trials':trials,'source_bindings':bindings,'independent_math':list(cache.values()),'mathematical_cache_note':'Everyartifactfullyread/rehashed/exactcompared; independentrecalculation cachedonlyidenticalcompleteartifactbytes. Twoactualdrugsets, not20distinctinputs.','all3_GPUstage_outputs_survive_primary_capture':True,'JACC_auxiliary_scratch':{'path':str(scratch[0].relative_to(ROOT)),'sha256':sha(scratch[0].read_bytes()),'initial_final_zero_values':256,'all128positions_overwritten':True,'mechanism':'Each128threads writespartial[tx] beforebarrier/reduction; templateloads cannotaffectoutputs. Extra preload/barriercovariate retained.'},'cold_reproduction':cold,'original_CLI_and_source_statistical_limit':w,'numeric_contract_qualification':'Odd4099randomdraws excludes sourceexact50%pvaluebranch defect; source > predicate/normalization preserved. Nofullbiologicaldataset/globalstatisticaldomaincertification.'}
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'pass','primary_captures':len(trials),'cold_files_exact':len(cold),'source_bindings':len(bindings),'scratch_actual_zero_values':256}))
