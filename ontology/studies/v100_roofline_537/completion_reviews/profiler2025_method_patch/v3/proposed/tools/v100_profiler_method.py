"""Private proposal: CPU-only exact NCU2025 method gate. Never invokes processes.
Root must create an approved receipt+bundle after successful saved calibration.
Historical audit needs only these frozen bytes, not today's SDK installation.
"""
import ast,base64,csv,hashlib,json,math,re,tempfile,types
from pathlib import Path
LEGACY_VERSION='NVIDIA (R) Nsight Compute Command Line Profiler\nCopyright (c) 2018-2023 NVIDIA Corporation\nVersion 2022.4.1.0 (build 32308335) (public-release)\n'
METHOD='ncu2025_2_0_V100_sm70_raw_v1'
UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
CONTROLLER='7211c35d2d5fc02e157b1142622cc31c6fd78510685e3b8244f15f5d675caf26'
MANIFEST='df47f5656c48ce113f126a92570a216fc50fe6352f939729e51a82e3a45241a0'

def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,msg):
 if not v:raise ValueError('Profiler method: '+msg)
def checked(data,digest):
 need(isinstance(digest,str)and re.fullmatch('[a-f0-9]{64}',digest),'missing/invalid SHA')
 need(sha(data)==digest,'SHA mismatch');return data
def _module(source,name):
 m=types.ModuleType(name);m.__file__=name;exec(compile(source,name,'exec'),m.__dict__);return m

def normalize_version_log(raw,expected_write_directories):
 """Remove ONLY one known LLguard line; retain exact remaining bytes/hashes.
 Calibration stores a merged log, not independently separated stdout/stderr.
 """
 first,sep,payload=raw.partition(b'\n')
 need(sep and first.startswith(b'V100_WRITE_GUARD '),'missing exact guard prefix')
 obj=json.loads(first[len(b'V100_WRITE_GUARD '):])
 need(set(obj)=={'landlock_abi','write_directories','privilege'},'unknown guard prefix schema')
 need(type(obj['landlock_abi'])is int and obj['landlock_abi']>=3,'invalid guard ABI')
 need(obj['write_directories']==expected_write_directories,'guard allowed paths differ')
 need(obj['privilege']=='existing unprivileged user; no_new_privs','guard policy mismatch')
 text=payload.decode('utf-8');lines=text.splitlines()
 need(len(lines)==3 and payload.endswith(b'\n'),'extra/missing version output')
 need(lines[0]=='NVIDIA (R) Nsight Compute Command Line Profiler','version header differs')
 need(re.fullmatch(r'Copyright \(c\) 2018-20[0-9]{2} NVIDIA Corporation',lines[1])is not None,'version copyright differs')
 need(re.fullmatch(r'Version [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ \(build [0-9]+\) \(public-release\)',lines[2])is not None,'version/build format differs')
 return {'payload':payload,'text':text,'payload_sha256':sha(payload),'merged_log_sha256':sha(raw),'guard_prefix_sha256':sha(first+b'\n'),'stream_scope':'exact version-shaped payload from merged log after one source-defined guard line; not separately captured stdout/stderr'}

def validate_saved(review_bytes,review_sha,bundle_bytes):
 """Validate saved root receipt and exact science; no current-file requirement."""
 checked(review_bytes,review_sha);r=json.loads(review_bytes)
 need(r.get('schema_version')==1 and r.get('method_id')==METHOD,'unrecognized method')
 need(r.get('status')=='root_approved_after_saved_calibration_pass','root adoption absent')
 checked(bundle_bytes,r['bundle_sha256']);packed=json.loads(bundle_bytes);files={}
 for name,item in packed['files'].items():
  need(not Path(name).is_absolute()and '..'not in Path(name).parts,'unsafe archive path')
  data=base64.b64decode(item['base64'],validate=True);checked(data,item['sha256']);files[name]=data
 need(sha(files['manifest.json'])==MANIFEST,'unexpected source manifest')
 need(sha(files['controller.py'])==CONTROLLER,'unexpected calibration controller')
 man=json.loads(files['manifest.json']);p=json.loads(files['provenance.json'])
 need(p['manifest']['sha256']==MANIFEST,'provenance/manifest join mismatch')
 need(sha(files['root_review_note.txt'])==r['root_review_note_sha256']and len(files['root_review_note.txt'].strip())>=32,'root note mismatch')
 need(len(man['bindings'])==83,'not exact83-binding experiment')
 need(p.get('status')=='new_profiler_calibration_pass_root_adoption_still_required'and bool(p.get('finished_at')),'calibration incomplete/failed')
 need(p.get('gpu_uuid')==UUID and p.get('registry_or_method_adopted')is False,'device/adoption mismatch')
 for key in ('source_tool_ELF_before','source_tool_ELF_final'):
  need(p[key]['all_unchanged']is True and p[key]['bindings']==man['bindings'],'calibration identity drift')
 identity=r['profiler']
 version_raw=files['precision/version.log'];need(version_raw.decode()==p['actual_profiler_version_text'],'full retained versionlog differs')
 counter_path=p['cases']['precision']['command'][p['cases']['precision']['command'].index('--log-file')+1]
 normalized=normalize_version_log(version_raw,[str(Path(counter_path).parent.parent)])
 need(normalized['payload']==files['precision/version.payload.bin'],'calibration normalized versionfile differs')
 need(p['actual_profiler_version_payload_text']==normalized['text']and p['actual_profiler_version_payload_sha256']==normalized['payload_sha256'],'002 normalized version provenance mismatch')
 vr=p['cases']['precision']['version_output_normalization']
 need(vr['raw_log_sha256']==normalized['merged_log_sha256']and vr['normalized_payload_sha256']==normalized['payload_sha256']and vr['exact_guard_preface_sha256']==normalized['guard_prefix_sha256'],'002 version normalizationreceipt mismatch')
 need(identity['version_stdout']==normalized['text'] and identity['version_stdout_sha256']==normalized['payload_sha256'] and identity['retained_version_log_sha256']==normalized['merged_log_sha256'],'normalized versionbytes differ')
 need(re.fullmatch(r'Version 2025\.2\.0\.0 \(build [0-9]+\) \(public-release\)',normalized['text'].splitlines()[2])is not None,'new method version differs')
 need(man['pinned_runtime_environment']=={'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'','LD_PRELOAD':''}and man['pinned_inherited_environment']=={},'unexpected calibration runtime')
 need(identity['executable']==man['ncu_executable']and identity['target_ELF']==man['ncu_target_ELF'],'profiler paths mismatch')
 for path,digestkey in ((identity['executable'],'executable_sha256'),(identity['target_ELF'],'target_ELF_sha256')):
  need(identity[digestkey]==man['bindings'][path],'profiler binary SHA mismatch')
 need(p['physical_half_lane_factor']==2 and p['physical_half_fma_ops_per_instruction']==4 and p['tensor_FLOPs_per_HMMA884_instruction']==512,'arithmetic factors changed')
 # Only pure source-count validation functions are extracted. Never controller.main.
 tree=ast.parse(files['controller.py']);names={'shape','key','rows_by_key','instr','positive_number','validate'}
 nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in names]
 need({x.name for x in nodes}==names,'pure calibration functions missing')
 ns={'re':re,'math':math};exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_calibration_validation','exec'),ns)
 need(sha(files['parser.py'])==man['bindings'][str(Path(man['source_root'])/'tools/v100_roofline_classify.py')],'calibration parser SHA mismatch')
 parser=_module(files['parser.py'],'frozen_method_parser');allrows=[]
 metrics=eval(compile(ast.parse(man['metrics_source_expression'],mode='eval'),'frozen_metric_inventory','eval'),{'__builtins__':{}},{})
 with tempfile.TemporaryDirectory(prefix='ncu-method-audit-')as temp:
  for case in ('precision','half_arithmetic'):
   cp=p['cases'][case];need(cp['status']=='calibration_case_pass','case incomplete')
   raw_identity=files[case+'/identity.log'];first,sep,payload=raw_identity.partition(b'\n')
   need(sep and first.startswith(b'V100_WRITE_GUARD '),'002 identity preface missing')
   guard=json.loads(first[len(b'V100_WRITE_GUARD '):]);counterfile=cp['command'][cp['command'].index('--log-file')+1]
   need(set(guard)=={'landlock_abi','write_directories','privilege'}and type(guard['landlock_abi'])is int and guard['landlock_abi']>=3 and guard['write_directories']==[str(Path(counterfile).parent.parent)]and guard['privilege']=='existing unprivileged user; no_new_privs','002 identity guard mismatch')
   ir=cp['identity_output_normalization'];need(sha(raw_identity)==ir['raw_log_sha256']and sha(first+b'\n')==ir['exact_guard_preface_sha256']and sha(payload)==ir['normalized_payload_sha256']and payload==files[case+'/identity.payload.bin'],'002 identity normalizationreceipt mismatch')
   lines=payload.decode().splitlines();need(len(lines)==1 and payload.endswith(b'\n'),'002 identity payload shape')
   row=next(csv.reader(lines));need(len(row)==2 and row[0].strip()==UUID and row[1].strip().lower()=='00000000:cb:00.0','002 identity UUID/PCI mismatch')
   need(cp['profile']['returncode']==0 and cp['profile']['stop_reason']is None,'case process failed')
   need(cp['source_bindings_before']['bindings']==man['bindings']and cp['source_tool_ELF_after']['bindings']==man['bindings'],'case83bindings mismatch')
   log=files[case+'/execution.log'].decode(errors='replace');line=man['cases'][case]['expected_host_check_line']
   need([x for x in log.splitlines()if x.startswith('calibration_correct=')]==[line]and cp['full_host_check']['pass']is True,'full host check missing')
   need(cp['command'][0]==identity['executable'],'calibration command executable mismatch')
   new=files[case+'/counters.csv'];old=files[case+'/old_counters.csv']
   need(sha(new)==cp['counters_sha256']==cp['new_counter_sha256'],'new counters SHA mismatch')
   need(sha(old)==cp['old_counter_sha256']==man['bindings'][man['cases'][case]['old_counters']],'old counters SHA mismatch')
   need(sha(files[case+'/source.cu'])==man['bindings'][man['cases'][case]['source']],'canary source SHA mismatch')
   for tag,data in [('new',new),('old',old)]:Path(temp,tag+'.csv').write_bytes(data)
   groups,_=parser.read_csv(Path(temp,'new.csv'));previous,_=parser.read_csv(Path(temp,'old.csv'))
   need(tuple(cp['expected_PCI'])==(0,203,0),'actual calibration PCI mismatch')
   rows=ns['validate'](case,groups,previous,metrics,(0,203,0))
   # JSON canonicalization compares tuple/list identities faithfully after serialization.
   need(json.dumps(rows,sort_keys=True)==json.dumps(cp['counter_comparisons'],sort_keys=True),'saved scientific validation differs')
   allrows.extend(rows)
 need(len(allrows)==9 and json.dumps(allrows,sort_keys=True)==json.dumps(p['calibration_rows'],sort_keys=True),'nine-row science missing')
 return {'method_id':METHOD,'review_sha256':review_sha,'bundle_sha256':sha(bundle_bytes),'calibration_provenance_sha256':sha(files['provenance.json']),'profiler':identity,'manifest':man}

def load(review_path,review_sha,executable,version_stdout=None):
 """Runtime preflight: validate archived science, then check live SDK files only."""
 path=Path(review_path).resolve();review=path.read_bytes();r=json.loads(checked(review,review_sha))
 bundle=(path.parent/r['bundle_file']).resolve();need(bundle.is_relative_to(path.parent),'bundle escapes receipt directory')
 raw=bundle.read_bytes();v=validate_saved(review,review_sha,raw);ident=v['profiler']
 need(str(Path(executable).resolve())==ident['executable'],'actual executable differs')
 if version_stdout is not None:need(version_stdout==ident['version_stdout'],'actual normalized version differs')
 sdk=str(Path(ident['executable']).parent)+'/'
 for name,digest in v['manifest']['bindings'].items():
  if name==ident['executable']or name.startswith(sdk):checked(Path(name).read_bytes(),digest)
 return v,review,raw

def pin_runtime(validated,env):
 result=dict(env);result.update(validated['manifest']['pinned_runtime_environment']);return result

def audit_profile(provenance,files):
 """New2025 profiles must archive method; historical2022 is unchanged."""
 m=provenance.get('profiler_method');version=provenance.get('profiler_version','')
 if not m:
  need(not version or version==LEGACY_VERSION,'unreviewed explicit profiler version without method');return None
 need(m['method_id']==METHOD,'unknown explicit method')
 v=validate_saved(files['profiler_method_review.json'],m['review_sha256'],files['profiler_method_bundle.json'])
 need(m['method_id']==v['method_id']and m['bundle_sha256']==v['bundle_sha256']and m['calibration_provenance_sha256']==v['calibration_provenance_sha256'],'method/calibration metadata mismatch')
 need(sha(files['profiler_method_validator.py'])==m['validator_sha256'],'validator source mismatch')
 preflight_raw=files['profiler_method_preflight.json'];need(sha(preflight_raw)==provenance['method_preflight_sha256'],'preflight sourcejoin mismatch')
 preflight=json.loads(preflight_raw);probe=preflight['version_process']
 need(preflight['status']=='passed'and probe['returncode']==0 and probe['stop_reason']is None and probe['log_sha256']==m['version_log_sha256'],'versionprocess receipt failed')
 argv=probe['executed_command'];need(argv[argv.index('--')+1:]==[v['profiler']['executable'],'--version'],'versionprobe argv mismatch')
 if provenance.get('profile'):
  argv=provenance['profile']['executed_command'];need(argv[argv.index('--')+1:]==provenance['command']and provenance['command'][0]==v['profiler']['executable'],'actualprofile executable/argv mismatch')
 normalized=normalize_version_log(files['profiler_version.log'],provenance['filesystem_write_guard']['write_directories'])
 need(normalized['payload']==files['profiler_version_payload.bin']and normalized['payload_sha256']==m['version_stdout_sha256']and normalized['merged_log_sha256']==m['version_log_sha256'],'profile versionreceipt mismatch')
 need(provenance['profiler_runtime_environment']==v['manifest']['pinned_runtime_environment'],'profile/calibration childenv differs')
 if provenance.get('status')=='profile_process_completed' or 'derived.json'in files:
  need('derived.json'in files,'completed method lacks derivedreport')
  op=provenance['operation_calibration'];need(op['sha256']==v['review_sha256']and op['half_lane_factor']==2 and op['calibration_provenance_sha256']==v['calibration_provenance_sha256'],'operation calibration notjoined')
  report=json.loads(files['derived.json']);need(report.get('profiler_method')==m,'derived methodjoin mismatch')
  need(all(k['scope_context'].get('profiler_method')==m and k['fp16_convention_evidence']==op and k['fp16_lane_factor']==2 for k in report['records']),'perkernel calibrationjoin mismatch')
 ident=v['profiler'];need(provenance['profiler_version']==ident['version_stdout']and provenance['profiler_executable']==ident['executable']and provenance['profiler_executable_sha256']==ident['executable_sha256'],'profile/calibration identity mismatch')
 return v
