"""Private proposal: CPU-only exact NCU2025 method gate. Never invokes processes.
Root must create an approved receipt+bundle after successful saved calibration.
Historical audit needs only these frozen bytes, not today's SDK installation.
"""
import ast,base64,hashlib,json,math,re,tempfile,types
from pathlib import Path
METHOD='ncu2025_2_0_V100_sm70_raw_v1'
UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
CONTROLLER='eaf532a0ac9ef114045fc7e12842191edcd667a29821eecace88f387d646a630'
MANIFEST='a28e3e67be537b85f77223379e6335abfba0c48e1e8648711928f0aeef27abae'

def sha(b):return hashlib.sha256(b).hexdigest()
def need(v,msg):
 if not v:raise ValueError('Profiler method: '+msg)
def checked(data,digest):
 need(isinstance(digest,str)and re.fullmatch('[a-f0-9]{64}',digest),'missing/invalid SHA')
 need(sha(data)==digest,'SHA mismatch');return data
def _module(source,name):
 m=types.ModuleType(name);m.__file__=name;exec(compile(source,name,'exec'),m.__dict__);return m

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
 identity=r['profiler'];need(identity['version_stdout']==p['actual_profiler_version_text']and 'Version 2025.2.0.0'in identity['version_stdout'],'version/build mismatch')
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

def load(review_path,review_sha,executable,version_stdout):
 """Runtime preflight: validate archived science, then check live SDK files only."""
 path=Path(review_path).resolve();review=path.read_bytes();r=json.loads(checked(review,review_sha))
 bundle=(path.parent/r['bundle_file']).resolve();need(bundle.is_relative_to(path.parent),'bundle escapes receipt directory')
 raw=bundle.read_bytes();v=validate_saved(review,review_sha,raw);ident=v['profiler']
 need(str(Path(executable).resolve())==ident['executable']and version_stdout==ident['version_stdout'],'actual profiler identity differs')
 sdk=str(Path(ident['executable']).parent)+'/'
 for name,digest in v['manifest']['bindings'].items():
  if name==ident['executable']or name.startswith(sdk):checked(Path(name).read_bytes(),digest)
 return v,review,raw

def audit_profile(provenance,files):
 """New2025 profiles must archive method; historical2022 is unchanged."""
 m=provenance.get('profiler_method');version=provenance.get('profiler_version','')
 if not m:
  need('Version 2025.'not in version,'2025 profile missing method receipt');return None
 need(m['method_id']==METHOD,'unknown explicit method')
 v=validate_saved(files['profiler_method_review.json'],m['review_sha256'],files['profiler_method_bundle.json'])
 need(m['method_id']==v['method_id']and m['bundle_sha256']==v['bundle_sha256'],'method metadata mismatch')
 if 'derived.json'in files:
  op=provenance['operation_calibration'];need(op['sha256']==v['review_sha256']and op['half_lane_factor']==2,'operation calibration notjoined')
 ident=v['profiler'];need(provenance['profiler_version']==ident['version_stdout']and provenance['profiler_executable']==ident['executable']and provenance['profiler_executable_sha256']==ident['executable_sha256'],'profile/calibration identity mismatch')
 return v
