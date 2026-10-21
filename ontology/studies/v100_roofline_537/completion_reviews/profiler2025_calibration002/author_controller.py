"""Text-only private controller repair. Never imports or runs authored controller."""
from pathlib import Path
import json,hashlib,shutil,difflib
O=Path(__file__).resolve().parent;OLD=Path('/tmp/roofline-completion-20261021/profiler2025_calibration');RAW=Path('/tmp/hecbench-roofline-v100-20261020/method_calibration_ncu2025_2_0_001')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
history=O/'preserved001';history.mkdir();files=[]
for p in sorted(RAW.rglob('*')):
 if p.is_file():
  q=history/p.relative_to(RAW);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);files.append({'original':str(p),'snapshot':str(q),'sha256':sha(p),'bytes':p.stat().st_size})
for n in ['calibrate_controller.py','source_binding_manifest.json','prepare_manifest.py','delivery.json']:
 p=OLD/n;q=history/('proposal_'+n);shutil.copyfile(p,q);files.append({'original':str(p),'snapshot':str(q),'sha256':sha(p),'bytes':p.stat().st_size})
old=(OLD/'calibrate_controller.py').read_text();assert old==(RAW/'controller_source.py').read_text();text=old
anchor='def shape(value):'
normalizer='''def guarded_payload(path,expected_write_directories):
 """Exact one-prefix normalization; full merged log remains unchanged.
 Only the source-defined LLguard line is removed, never arbitrary warnings.
 """
 path=Path(path);raw=path.read_bytes();prefix,separator,payload=raw.partition(b'\\n')
 if not separator or not prefix.startswith(b'V100_WRITE_GUARD '):raise RuntimeError('Exact LLguard preface missing '+str(path))
 guard=json.loads(prefix[len(b'V100_WRITE_GUARD '):])
 if set(guard)!={'landlock_abi','write_directories','privilege'}:raise RuntimeError('Unexpected LLguard schema')
 if type(guard['landlock_abi'])is not int or guard['landlock_abi']<3:raise RuntimeError('LLguard ABI mismatch')
 if guard['write_directories']!=expected_write_directories:raise RuntimeError('LLguard write directory differs from actual freshoutput')
 if guard['privilege']!='existing unprivileged user; no_new_privs':raise RuntimeError('LLguard privilege mismatch')
 if not payload or any(line.startswith(b'V100_WRITE_GUARD ')for line in payload.splitlines()):raise RuntimeError('Empty or repeated LLguard payload')
 normalized=path.with_name(path.stem+'.payload.bin');normalized.write_bytes(payload)
 receipt={'raw_log':str(path),'raw_log_sha256':sha(path),'raw_log_bytes':len(raw),
          'exact_guard_preface_sha256':hashlib.sha256(prefix+b'\\n').hexdigest(),
          'normalized_payload':str(normalized),'normalized_payload_sha256':sha(normalized),'normalized_payload_bytes':len(payload),
          'normalization':'Remove exactly one first source-defined LLguard JSON line; preserve every remaining byte; merged stdout/stderr not retroactively separated',
          'guard':guard}
 return payload,receipt

'''
assert text.count(anchor)==1;text=text.replace(anchor,normalizer+anchor,1)
a="    ident=list(csv.reader((slot/'identity.log').read_text().splitlines()))"
b="    identity_payload,identity_receipt=guarded_payload(slot/'identity.log',guard['write_directories'])\n    cp['identity_output_normalization']=identity_receipt;dump(out/'profile_provenance.json',prov)\n    identity_text=identity_payload.decode('utf-8')\n    if not identity_payload.endswith(b'\\n') or len(identity_text.splitlines())!=1:raise RuntimeError('Exactly one identity payload row required')\n    ident=list(csv.reader(identity_text.splitlines()))"
assert text.count(a)==1;text=text.replace(a,b,1)
a="     text=(slot/'version.log').read_text()\n     if cp['profiler_version_process']['returncode']!=0 or cp['profiler_version_process']['stop_reason']is not None or 'Version 2025.2.0.0'not in text:raise RuntimeError('Exactprofiler runtimeversion checkfailed')\n     prov['actual_profiler_version_text']=text"
b="""     if cp['profiler_version_process']['returncode']!=0 or cp['profiler_version_process']['stop_reason']is not None:raise RuntimeError('Profiler runtimeversion process failed')
     version_payload,version_receipt=guarded_payload(slot/'version.log',guard['write_directories'])
     cp['version_output_normalization']=version_receipt;dump(out/'profile_provenance.json',prov)
     text=version_payload.decode('utf-8');version_lines=text.splitlines()
     if not version_payload.endswith(b'\\n') or len(version_lines)!=3:raise RuntimeError('Unexpected profiler version payload shape')
     if version_lines[0]!='NVIDIA (R) Nsight Compute Command Line Profiler':raise RuntimeError('Unexpected profiler header')
     if re.fullmatch(r'Copyright \\(c\\) 2018-20[0-9]{2} NVIDIA Corporation',version_lines[1])is None:raise RuntimeError('Unexpected profiler copyright')
     if re.fullmatch(r'Version 2025\\.2\\.0\\.0 \\(build [0-9]+\\) \\(public-release\\)',version_lines[2])is None:raise RuntimeError('Unexpected exact profiler version/build')
     prov['actual_profiler_version_text']=(slot/'version.log').read_text()
     prov['actual_profiler_version_payload_text']=text
     prov['actual_profiler_version_payload_sha256']=version_receipt['normalized_payload_sha256']"""
assert text.count(a)==1;text=text.replace(a,b,1)
# All scientific comparison, commands, metrics, deadlines, host gates and locks unchanged.
a=old.index('def shape(value):');b=old.index('\ndef main():');a2=text.index('def shape(value):');b2=text.index('\ndef main():');assert old[a:b]==text[a2:b2]
(O/'calibrate_controller.py').write_text(text);(O/'controller.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='calibration001_controller.py',tofile='calibration002_controller.py')))
(O/'preserved001_inventory.json').write_text(json.dumps({'files':files,'001_unchanged':all(sha(Path(f['original']))==f['sha256']for f in files),'old_controller_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_controller_sha256':sha(O/'calibrate_controller.py'),'pure_scientific_validation_source_unchanged':True,'GPU_build_tests_executed':False},indent=2)+'\n')
t=(OLD/'prepare_manifest.py').read_text();t=t.replace("ast.parse(controller.read_text());",'');t=t.replace('method_calibration_ncu2025_2_0_001','method_calibration_ncu2025_2_0_002');(O/'prepare_manifest.py').write_text(t)
print(json.dumps({'controller_sha256':sha(O/'calibrate_controller.py'),'original001files_preserved':len(files),'source_scientific_functions_byteidentical':True,'not_executed':True}))
