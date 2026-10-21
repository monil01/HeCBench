from pathlib import Path
import hashlib,json,csv,datetime
D=Path(__file__).resolve().parent;R=Path('/tmp/hecbench-roofline-v100-20261020');S=Path('/home/imo/HeCBench_julia')
paths=[Path('/usr/share/doc/nsight-compute/NsightComputeCli/index.html'),Path('/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Compute/docs/NsightComputeCli/index.html')]
paths += [R/'batch003-clink/clink/ka'/n for n in ['plan.json','profile_provenance.json','counters.csv','execution.log','collector_source.py','parser_source.py']]
paths += [R/'selector_probe'/n for n in ['selector.cu','selector_shapes.cu','shape_combined_provenance.json','shape_combined.csv','shape_combined.log','per_config_tmp2_provenance.json','per_config_tmp2.csv']]
paths += [S/'src/clink-julia'/n for n in ['clink-julia-KA/main.jl','study_common.jl','study_device.jl']]
def sha(x):return hashlib.sha256(x).hexdigest()
raw={str(p):p.read_bytes()for p in paths};bindings={p:{'sha256':sha(x),'bytes':len(x)}for p,x in raw.items()}
excerpts=[]
for p,a,b in [(paths[0],975,1044),(paths[0],1075,1095),(paths[1],936,968),(paths[1],983,1005)]:
 text=raw[str(p)].decode();lines=text.splitlines();excerpts.append({'path':str(p),'source_sha256':sha(raw[str(p)]),'start_line':a,'end_line':b,'exact_lines':'\n'.join(lines[a-1:b])})
(D/'installed_primary_document_excerpts.json').write_text(json.dumps(excerpts,indent=2)+'\n')
rows={}
for p in [R/'selector_probe/shape_combined.csv',R/'selector_probe/per_config_tmp2.csv',R/'batch003-clink/clink/ka/counters.csv']:
 lines=raw[str(p)].decode().splitlines();start=next(i for i,l in enumerate(lines)if l.startswith('"ID"'));rr=list(csv.DictReader(lines[start:]));rows[str(p)]=[{k:x.get(k)for k in ['ID','Kernel Name','Grid Size','Block Size','Context','Stream']}for x in rr if x.get('ID','').isdigit()]
prov=json.loads(raw[str(R/'batch003-clink/clink/ka/profile_provenance.json')]);plan=json.loads(raw[str(R/'batch003-clink/clink/ka/plan.json')]);command=prov['command'];assert command[command.index('--kernel-id')+1]=='::regex:.*:1'
report={'status':'CPU_PRIMARY_DOC_AND_SAVED_EVIDENCE_REVIEW_NO_RUNTIME_TEST','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bindings':bindings,'observed_raw_rows':rows,'clink':{'status':prov['status'],'profile':prov['profile'],'command':command,'configuration_id':plan['configuration_id'],'selected_contract_fields':{k:plan.get(k)for k in ['reference_result_sha256','raw_config_sha256','execution_bindings']},'sampled_semantic_family':'unknown','authored_kernel_identity_not_inferred_from_signature':True},'conclusion':'Historical2022 documented intent supports per-function invocation plus per-launch-config partitioning. Neither installed version documents universal aggregation of distinct function names solely by geometry. Saved native probe provides limited empirical counterexample to such blanket aggregation. No document/probe proves campaign family completeness or resolves sampled clink signature.','qualification':'Probe command omits explicit kernel-name-base mangled and does not freeze exact executed profiler/toolchain identities in its small provenance; treat as limited retained observation, not exact full-method calibration or proof of every Julia specialization. No new run performed.','proposed_method_scope':'first matching invocation under the recorded wildcard kernel-id and per-launch-config filter; per-function/configuration sampling intent, observed saved launch set only; no enumeration/completeness guarantee','future_version_distinction':'2025 wording additionally specifies kernel name, grid/block/shared and optional context/stream for invocation count; do not use newer wording as retroactive proof of2022 implementation details.','acceptance_unchanged':True,'KA_family_hold_retained':True,'gpu_runs':0,'builds':0,'tests':0,'shared_mutations':0}
for p,x in raw.items():
 if Path(p).read_bytes()!=x:raise RuntimeError('Input changed during review '+p)
report['all_bindings_before_after_stable']=True
(D/'review.json').write_text(json.dumps(report,indent=2)+'\n')
