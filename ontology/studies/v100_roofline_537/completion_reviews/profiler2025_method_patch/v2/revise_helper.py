from pathlib import Path
O=Path(__file__).parent;p=O/'proposed/tools/v100_profiler_method.py';t=p.read_text()
a=t.index('def validate_saved(')
t=t[:a]+'''def normalize_version_log(raw,expected_write_directories):
 """Remove ONLY one known LLguard line; retain exact remaining bytes/hashes.
 Calibration stores a merged log, not independently separated stdout/stderr.
 """
 first,sep,payload=raw.partition(b'\\n')
 need(sep and first.startswith(b'V100_WRITE_GUARD '),'missing exact guard prefix')
 obj=json.loads(first[len(b'V100_WRITE_GUARD '):])
 need(set(obj)=={'landlock_abi','write_directories','privilege'},'unknown guard prefix schema')
 need(type(obj['landlock_abi'])is int and obj['landlock_abi']>=3,'invalid guard ABI')
 need(obj['write_directories']==expected_write_directories,'guard allowed paths differ')
 need(obj['privilege']=='existing unprivileged user; no_new_privs','guard policy mismatch')
 text=payload.decode('utf-8');lines=text.splitlines()
 need(len(lines)==3 and payload.endswith(b'\\n'),'extra/missing version output')
 need(lines[0]=='NVIDIA (R) Nsight Compute Command Line Profiler','version header differs')
 need(re.fullmatch(r'Copyright \\(c\\) 2018-20[0-9]{2} NVIDIA Corporation',lines[1])is not None,'version copyright differs')
 need(re.fullmatch(r'Version [0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+ \\(build [0-9]+\\) \\(public-release\\)',lines[2])is not None,'version/build format differs')
 return {'payload':payload,'text':text,'payload_sha256':sha(payload),'merged_log_sha256':sha(raw),'guard_prefix_sha256':sha(first+b'\\n'),'stream_scope':'exact version-shaped payload from merged log after one source-defined guard line; not separately captured stdout/stderr'}

'''+t[a:]
t=t.replace("identity=r['profiler'];need(identity['version_stdout']==p['actual_profiler_version_text']and 'Version 2025.2.0.0'in identity['version_stdout'],'version/build mismatch')", "identity=r['profiler']\n version_raw=files['precision/version.log'];need(version_raw.decode()==p['actual_profiler_version_text'],'full retained versionlog differs')\n counter_path=p['cases']['precision']['command'][p['cases']['precision']['command'].index('--log-file')+1]\n normalized=normalize_version_log(version_raw,[str(Path(counter_path).parent.parent)])\n need(identity['version_stdout']==normalized['text'] and identity['version_stdout_sha256']==normalized['payload_sha256'] and identity['retained_version_log_sha256']==normalized['merged_log_sha256'],'normalized versionbytes differ')\n need(re.fullmatch(r'Version 2025\\.2\\.0\\.0 \\(build [0-9]+\\) \\(public-release\\)',normalized['text'].splitlines()[2])is not None,'new method version differs')\n need(man['pinned_runtime_environment']=={'PATH':'/usr/bin:/bin','LD_LIBRARY_PATH':'','LD_PRELOAD':''}and man['pinned_inherited_environment']=={},'unexpected calibration runtime')")
t=t.replace('def load(review_path,review_sha,executable,version_stdout):','def load(review_path,review_sha,executable,version_stdout=None):')
t=t.replace("need(str(Path(executable).resolve())==ident['executable']and version_stdout==ident['version_stdout'],'actual profiler identity differs')", "need(str(Path(executable).resolve())==ident['executable'],'actual executable differs')\n if version_stdout is not None:need(version_stdout==ident['version_stdout'],'actual normalized version differs')")
t=t.replace("return v,review,raw\n\ndef audit_profile", "return v,review,raw\n\ndef pin_runtime(validated,env):\n result=dict(env);result.update(validated['manifest']['pinned_runtime_environment']);return result\n\ndef audit_profile")
t=t.replace("need(m['method_id']==v['method_id']and m['bundle_sha256']==v['bundle_sha256'],'method metadata mismatch')", "need(m['method_id']==v['method_id']and m['bundle_sha256']==v['bundle_sha256']and m['calibration_provenance_sha256']==v['calibration_provenance_sha256'],'method/calibration metadata mismatch')\n need(sha(files['profiler_method_validator.py'])==m['validator_sha256'],'validator source mismatch')\n normalized=normalize_version_log(files['profiler_version.log'],provenance['filesystem_write_guard']['write_directories'])\n need(normalized['payload']==files['profiler_version_payload.bin']and normalized['payload_sha256']==m['version_stdout_sha256']and normalized['merged_log_sha256']==m['version_log_sha256'],'profile versionreceipt mismatch')\n need(provenance['profiler_runtime_environment']==v['manifest']['pinned_runtime_environment'],'profile/calibration childenv differs')")
t=t.replace("if 'derived.json'in files:\n  op=provenance['operation_calibration'];need(op['sha256']==v['review_sha256']and op['half_lane_factor']==2,'operation calibration notjoined')", "if provenance.get('status')=='profile_process_completed' or 'derived.json'in files:\n  need('derived.json'in files,'completed method lacks derivedreport')\n  op=provenance['operation_calibration'];need(op['sha256']==v['review_sha256']and op['half_lane_factor']==2 and op['calibration_provenance_sha256']==v['calibration_provenance_sha256'],'operation calibration notjoined')\n  report=json.loads(files['derived.json']);need(report.get('profiler_method')==m,'derived methodjoin mismatch')\n  need(all(k['scope_context'].get('profiler_method')==m and k['fp16_convention_evidence']==op and k['fp16_lane_factor']==2 for k in report['records']),'perkernel calibrationjoin mismatch')")
p.write_text(t)
p=O/'pack_method.py';t=p.read_text();t=t.replace("'root_review_note.txt':note}","'root_review_note.txt':note,'precision/version.log':(root/'precision/version.log').read_bytes()}")
t=t.replace("ident={'executable':", "normalized=gate.normalize_version_log(f['precision/version.log'],[str(root)])\n ident={'version_stdout_sha256':normalized['payload_sha256'],'retained_version_log_sha256':normalized['merged_log_sha256'],'executable':")
t=t.replace("'version_stdout':pr['actual_profiler_version_text']", "'version_stdout':normalized['text']")
p.write_text(t)
