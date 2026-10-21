"""Author source patch only. Does not import/run proposed target code."""
from pathlib import Path
import hashlib,json,difflib
W=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;original={};modified={}
for name in ['v100_roofline_collect.py','v100_roofline_summarize.py','v100_roofline_audit.py']:
 p=W/'tools'/name;original[name]=p.read_text();modified[name]=original[name]
assert hashlib.sha256(original['v100_roofline_collect.py'].encode()).hexdigest()=='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
def sub(name,old,new):
 t=modified[name];assert t.count(old)==1,(name,old[:80],t.count(old));modified[name]=t.replace(old,new,1)
c='v100_roofline_collect.py'
sub(c,'isolated_assertion_reviewed=False):','isolated_assertion_reviewed=False, profiler_executable=None, profiler_method_review=None, profiler_method_review_sha256=None):')
sub(c,"    command = list(plan['profile_target_command'])",'''    # New method is opt-in and fail-closed before any target build/profile.
    ncu_path = str(Path(profiler_executable or shutil.which('ncu')).resolve())
    version_text = subprocess.run([ncu_path,'--version'],env=env,capture_output=True,text=True,check=True).stdout
    method = None
    if profiler_method_review:
        import v100_profiler_method as method_gate
        method, method_review_raw, method_bundle_raw = method_gate.load(
            profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)
        (slot/'profiler_method_review.json').write_bytes(method_review_raw)
        (slot/'profiler_method_bundle.json').write_bytes(method_bundle_raw)
        method_source=Path(method_gate.__file__).read_bytes()
        (slot/'profiler_method_validator.py').write_bytes(method_source)
        method_label={k:method[k] for k in ('method_id','review_sha256','bundle_sha256','calibration_provenance_sha256')}
        method_label['validator_sha256']=hashlib.sha256(method_source).hexdigest()
    elif 'Version 2025.' in version_text:
        raise RuntimeError('NCU2025 requires explicit passed/root-approved method receipt and SHA')
    elif profiler_executable or profiler_method_review_sha256:
        raise RuntimeError('Explicit alternate profiler requires method receipt')
    command = list(plan['profile_target_command'])''')
sub(c,"    ncu = ['ncu', '--target-processes', 'all']", "    ncu = [ncu_path, '--target-processes', 'all']")
sub(c,"    (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)", "    if method:\n        provenance.update(profiler_method=method_label,profiler_version=version_text,profiler_executable=ncu_path,profiler_executable_sha256=sha(ncu_path))\n    (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)")
sub(c,"            provenance['profiler_version'] = subprocess.run(['ncu', '--version'],\n                capture_output=True, text=True, check=True).stdout\n            provenance['profiler_executable'] = str(Path(shutil.which('ncu')).resolve())", "            provenance['profiler_version'] = version_text\n            provenance['profiler_executable'] = ncu_path\n            if method:\n                provenance['profiler_method']=method_label")
sub(c,"            bindings(plan)\n            if isolated_review:", "            bindings(plan)\n            if method:\n                method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)\n            if isolated_review:")
sub(c,"                    context.update(gpu_sm70=True,", "                    if method:\n                        prior_calibration=provenance['operation_calibration']\n                        provenance['operation_calibration']={**method_label,'path':str(Path(profiler_method_review).resolve()),'sha256':method['review_sha256'],'half_lane_factor':2,'default_tensor_conversion_enabled':False,'prior_source_mode_calibration':prior_calibration}\n                        context.update(profiler_method=method_label)\n                    context.update(gpu_sm70=True,")
sub(c,"                    dump(slot/'derived.json', derived)", "                    if method:derived['profiler_method']=method_label\n                    dump(slot/'derived.json', derived)")
sub(c,"    ap.add_argument('--selection-overrides',type=Path)", "    ap.add_argument('--profiler-executable',type=Path)\n    ap.add_argument('--profiler-method-review',type=Path)\n    ap.add_argument('--profiler-method-review-sha256')\n    ap.add_argument('--selection-overrides',type=Path)")
sub(c,"a.profile_window_reviewed,a.isolated_assertion_reviewed)","a.profile_window_reviewed,a.isolated_assertion_reviewed,a.profiler_executable,a.profiler_method_review,a.profiler_method_review_sha256)")
s='v100_roofline_summarize.py'
sub(s,"'negative_prechecks.json','positive_profile_outcome.json','joined_execution_contract.json',", "'negative_prechecks.json','positive_profile_outcome.json','joined_execution_contract.json',\n                 'profiler_method_review.json','profiler_method_bundle.json','profiler_method_validator.py',")
sub(s,"        if not prov.get('finished_at'):\n            continue", "        if not prov.get('finished_at'):\n            continue\n        if prov.get('profiler_method') or 'Version 2025.' in prov.get('profiler_version',''):\n            import v100_profiler_method as method_gate\n            names=('profiler_method_review.json','profiler_method_bundle.json')\n            method_gate.audit_profile(prov,{n:(provpath.parent/n).read_bytes() for n in names})")
sub(s,"                 'selector_scope':prov['selector_scope']}","                 'selector_scope':prov['selector_scope'],\n                 'profiler_method':prov.get('profiler_method'),\n                 'profiler_version':prov.get('profiler_version')}")
sub(s,"        state.update(profiling_status=effective_status,", "        state.update(profiler_method=prov.get('profiler_method'),\n            profiler_version=prov.get('profiler_version'),profiling_status=effective_status,")
sub(s,"                'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',", "                'profiler_method_id':(state.get('profiler_method')or{}).get('method_id','legacy2022_recorded_or_identity_incomplete'),\n                'profiler_version':state.get('profiler_version'),\n                'method_review_sha256':(state.get('profiler_method')or{}).get('review_sha256'),\n                'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',")
sub(s,"                common={'benchmark':ledger['benchmark'],'model':model,", "                common={'profiler_method_id':slots[-1]['profiler_method_id'],\n                        'profiler_version':slots[-1]['profiler_version'],\n                        'method_review_sha256':slots[-1]['method_review_sha256'],\n                        'benchmark':ledger['benchmark'],'model':model,")
sub(s,"                    points.append({'benchmark':ledger['benchmark'],'model':model,", "                    points.append({'profiler_method_id':slots[-1]['profiler_method_id'],\n                        'profiler_version':slots[-1]['profiler_version'],\n                        'method_review_sha256':slots[-1]['method_review_sha256'],\n                        'benchmark':ledger['benchmark'],'model':model,")
a='v100_roofline_audit.py'
sub(a,"                if 'derived.json' in files:",'''                if prov.get('profiler_method') or 'Version 2025.' in prov.get('profiler_version',''):
                    method_source=files['profiler_method_validator.py']
                    require(sha(method_source)==prov['profiler_method']['validator_sha256'],'Method validator SHA mismatch')
                    method_module=types.ModuleType('frozen_profiler_method')
                    exec(compile(method_source,'frozen_profiler_method','exec'),method_module.__dict__)
                    method_module.audit_profile(prov,files)
                    row['profiler_method']=prov['profiler_method']
                if 'derived.json' in files:''')
sub(a,"                    require(report['input_sha256'] == sha(files['counters.csv']), 'Derivation input SHA mismatch')", "                    require(report['input_sha256'] == sha(files['counters.csv']), 'Derivation input SHA mismatch')\n                    if prov.get('profiler_method'):\n                        require(report.get('profiler_method')==prov['profiler_method'],'Derived method mismatch')\n                        require(all(x.get('scope_context',{}).get('profiler_method')==prov['profiler_method'] for x in report['records']),'Kernel method mismatch')")
patch=[];bindings=[]
for name,text in modified.items():
 p=O/'proposed/tools'/name;p.write_text(text);patch.extend(difflib.unified_diff(original[name].splitlines(True),text.splitlines(True),fromfile='a/tools/'+name,tofile='b/tools/'+name));bindings.append({'file':'tools/'+name,'base_sha256':hashlib.sha256(original[name].encode()).hexdigest(),'proposed_sha256':hashlib.sha256(text.encode()).hexdigest()})
helper=O/'proposed/tools/v100_profiler_method.py';patch.extend(difflib.unified_diff([],helper.read_text().splitlines(True),fromfile='/dev/null',tofile='b/tools/v100_profiler_method.py'));(O/'minimal.patch').write_text(''.join(patch));(O/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');print(json.dumps(bindings,indent=2))
