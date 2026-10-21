"""Text-only authoring: never imports/runs proposed helper/collector/tests."""
from pathlib import Path
import hashlib,json,difflib
W=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent;names=['v100_roofline_collect.py','v100_roofline_summarize.py','v100_roofline_audit.py'];original={n:(W/'tools'/n).read_text()for n in names};modified=dict(original)
assert hashlib.sha256(original[names[0]].encode()).hexdigest()=='32a9cfeba983dc3ae8e5d9b8bb85ca4e7736821be0a9c206bb57b80c7471da4c'
assert hashlib.sha256(original[names[1]].encode()).hexdigest()=='505d98ab675427156abed15d8796931a78abea9e6c1d950679a7a25aedc1c802'
assert hashlib.sha256(original[names[2]].encode()).hexdigest()=='28faa61716a177539c4854bbcaae64c94e7739a5a1ef43bcfbe72ee1798b86f3'
def sub(n,a,b):
 assert modified[n].count(a)==1,(n,a[:70],modified[n].count(a));modified[n]=modified[n].replace(a,b,1)
c=names[0]
sub(c,'isolated_assertion_reviewed=False):','isolated_assertion_reviewed=False, profiler_executable=None, profiler_method_review=None, profiler_method_review_sha256=None):')
sub(c,"    command = list(plan['profile_target_command'])",'''    # Source-only method rejection is durable and uses zero target/GPU/build runs.
    method = None
    preflight={'status':'started','started_at':utc(),'new_gpu_processes':0,'new_build_processes':0,
               'requested_profiler':str(profiler_executable) if profiler_executable else None,
               'review':str(profiler_method_review) if profiler_method_review else None,
               'review_sha256':profiler_method_review_sha256}
    try:
        import v100_profiler_method as method_gate
        candidate=profiler_executable or shutil.which('ncu')
        if not candidate:raise RuntimeError('Profiler executable unavailable')
        ncu_path=str(Path(candidate).resolve())
        if profiler_method_review:
            method,method_review_raw,method_bundle_raw=method_gate.load(
                profiler_method_review,profiler_method_review_sha256,ncu_path)
            env=method_gate.pin_runtime(method,env)
            (slot/'profiler_method_review.json').write_bytes(method_review_raw)
            (slot/'profiler_method_bundle.json').write_bytes(method_bundle_raw)
        elif profiler_executable or profiler_method_review_sha256:
            raise RuntimeError('Explicit alternate profiler requires reviewed method receipt and SHA')
        method_source=Path(method_gate.__file__).read_bytes()
        (slot/'profiler_method_validator.py').write_bytes(method_source)
        preflight['version_process']=run([ncu_path,'--version'],env,scratch,slot/'profiler_version.log',
            60,reserve_mib*1024**2,guard,monitor_gpu=False)
        if preflight['version_process']['returncode']!=0 or preflight['version_process']['stop_reason'] is not None:
            raise RuntimeError('Profiler version process failed')
        normalized=method_gate.normalize_version_log((slot/'profiler_version.log').read_bytes(),guard['write_directories'])
        version_text=normalized['text']
        (slot/'profiler_version_payload.bin').write_bytes(normalized['payload'])
        preflight.update(version_stdout_sha256=normalized['payload_sha256'],version_log_sha256=normalized['merged_log_sha256'])
        if method:
            method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)
            method_label={k:method[k]for k in ('method_id','review_sha256','bundle_sha256','calibration_provenance_sha256')}
            method_label.update(validator_sha256=hashlib.sha256(method_source).hexdigest(),
                version_stdout_sha256=normalized['payload_sha256'],version_log_sha256=normalized['merged_log_sha256'])
            # Absolute target Julia, no PATH/LD fallback when pinned tool runtime fails.
            command=[str(julia/'bin/julia') if x=='julia' or x=='/home/imo/.juliaup/bin/julia' else x for x in plan['profile_target_command']]
            for value in command:
                for key,expected in method['manifest']['pinned_runtime_environment'].items():
                    if value.startswith(key+'=') and value!=key+'='+expected:
                        raise RuntimeError('Target command overrides pinned tool runtime '+key)
        else:
            legacy=json.loads((root/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json').read_bytes())
            if version_text!=legacy['tool_versions']['ncu']:
                raise RuntimeError('Default method admits only exact saved legacy2022 version/build stdout')
            command=list(plan['profile_target_command'])
        preflight.update(status='passed',finished_at=utc())
        dump(slot/'profiler_method_preflight.json',preflight)
    except Exception as error:
        preflight.update(status='rejected',error=repr(error),finished_at=utc())
        dump(slot/'profiler_method_preflight.json',preflight)
        hold={'schema_version':1,'started_at':preflight['started_at'],'finished_at':utc(),
            'benchmark':plan['benchmark'],'model':plan['model'],'configuration_id':plan['configuration_id'],
            'status':'not_started_profiler_method_preflight_hold','gpu_uuid':UUID,
            'plan_sha256':sha(plan_path),'collector_sha256':COLLECTOR_SHA256,
            'command':[],'selector_scope':'No target/profile launch; method preflight rejected',
            'new_gpu_processes':0,'new_build_processes':0,
            'method_preflight_sha256':sha(slot/'profiler_method_preflight.json')}
        (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)
        dump(slot/'profile_provenance.json',hold)
        return {'benchmark':plan['benchmark'],'model':plan['model'],'status':hold['status'],'error':repr(error)}''')
sub(c,"    ncu = ['ncu', '--target-processes', 'all']", "    ncu = [ncu_path, '--target-processes', 'all']")
sub(c,"    (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)\n    dump(slot/'profile_provenance.json', provenance)","    if method:\n        provenance.update(profiler_method=method_label,profiler_runtime_environment={k:env[k]for k in method['manifest']['pinned_runtime_environment']},profiler_version=version_text,profiler_executable=ncu_path,profiler_executable_sha256=sha(ncu_path))\n    provenance['method_preflight_sha256']=sha(slot/'profiler_method_preflight.json')\n    (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)\n    dump(slot/'profile_provenance.json', provenance)")
sub(c,"            provenance['profiler_version'] = subprocess.run(['ncu', '--version'],\n                capture_output=True, text=True, check=True).stdout\n            provenance['profiler_executable'] = str(Path(shutil.which('ncu')).resolve())", "            provenance['profiler_version'] = version_text\n            provenance['profiler_executable'] = ncu_path")
sub(c,"            provenance['profile'] = run(ncu, env, scratch, slot/'execution.log',", "            if method:\n                method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)\n                if {k:env[k]for k in method['manifest']['pinned_runtime_environment']}!=method['manifest']['pinned_runtime_environment']:raise RuntimeError('Pinned runtime drift')\n            provenance['profile'] = run(ncu, env, scratch, slot/'execution.log',")
sub(c,"            bindings(plan)\n            if isolated_review:","            bindings(plan)\n            if method:method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)\n            if isolated_review:")
sub(c,"                    context.update(gpu_sm70=True,", "                    if method:\n                        prior_calibration=provenance['operation_calibration']\n                        provenance['operation_calibration']={**method_label,'path':str(Path(profiler_method_review).resolve()),'sha256':method['review_sha256'],'half_lane_factor':2,'default_tensor_conversion_enabled':False,'prior_source_mode_calibration':prior_calibration}\n                        context.update(profiler_method=method_label)\n                    context.update(gpu_sm70=True,")
sub(c,"                    dump(slot/'derived.json', derived)", "                    if method:derived['profiler_method']=method_label\n                    dump(slot/'derived.json', derived)")
sub(c,"    ap.add_argument('--selection-overrides',type=Path)", "    ap.add_argument('--profiler-executable',type=Path)\n    ap.add_argument('--profiler-method-review',type=Path)\n    ap.add_argument('--profiler-method-review-sha256')\n    ap.add_argument('--selection-overrides',type=Path)")
sub(c,"a.profile_window_reviewed,a.isolated_assertion_reviewed)","a.profile_window_reviewed,a.isolated_assertion_reviewed,a.profiler_executable,a.profiler_method_review,a.profiler_method_review_sha256)")
s=names[1]
sub(s,'import fcntl\n','import fcntl\nfrom v100_profiler_method import LEGACY_VERSION\n')
sub(s,"'negative_prechecks.json','positive_profile_outcome.json','joined_execution_contract.json',", "'negative_prechecks.json','positive_profile_outcome.json','joined_execution_contract.json',\n                 'profiler_method_review.json','profiler_method_bundle.json','profiler_method_validator.py',\n                 'profiler_method_preflight.json','profiler_version.log','profiler_version_payload.bin',")
sub(s,"        if not prov.get('finished_at'):\n            continue",'''        if not prov.get('finished_at'):
            continue
        method_error = None
        if prov.get('status')=='not_started_profiler_method_preflight_hold':
            method_error='Retained preflight rejection; no target/GPU/build execution'
        elif prov.get('profiler_method') or (prov.get('profiler_version') and prov['profiler_version']!=LEGACY_VERSION):
            try:
                import v100_profiler_method as method_gate
                names=('profiler_method_review.json','profiler_method_bundle.json','profiler_method_validator.py',
                       'profiler_version.log','profiler_version_payload.bin','profiler_method_preflight.json','derived.json')
                available={n:(provpath.parent/n).read_bytes() for n in names if (provpath.parent/n).is_file()}
                method_gate.audit_profile(prov,available)
            except Exception as error:
                method_error=repr(error)''')
sub(s,"                 'selector_scope':prov['selector_scope']}","                 'selector_scope':prov['selector_scope'],\n                 'raw_process_status':prov['status'],\n                 'profiler_method':prov.get('profiler_method'),\n                 'profiler_version':prov.get('profiler_version')}\n        if method_error:\n            attempt.update(status='profile_method_unapproved',method_validation_error=method_error)")
sub(s,"        observation=slot/'derived.json'", "        if method_error:\n            for saved_attempt in attempts:\n                if saved_attempt['profile_id']==attempt['profile_id']:saved_attempt.update(attempt)\n            if str(target.relative_to(root)) in state.get('profile_ids',[]):\n                state.update(profiling_status='profile_method_unapproved',profile_ids=[],sampled_kernel_count=0,sampled_kernel_precision_regimes={},sampled_kernel_predicted_resource_limits={},roi_coverage='method_unapproved_no_positioning',roofline_position=None,roofline_region=None,observed_bottleneck=None)\n                state.pop('kernel_observations',None)\n            dump(path,ledger)\n            continue\n        observation=slot/'derived.json'")
sub(s,"        state.update(profiling_status=effective_status,", "        state.update(profiler_method=prov.get('profiler_method'),\n            profiler_version=prov.get('profiler_version'),profiling_status=effective_status,")
sub(s,"                'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',", "                'profiler_method_id':(state.get('profiler_method')or{}).get('method_id','legacy2022_recorded_or_identity_incomplete'),\n                'profiler_version':state.get('profiler_version'),\n                'method_review_sha256':(state.get('profiler_method')or{}).get('review_sha256'),\n                'gpu_uuid':'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d',")
sub(s,"                common={'benchmark':ledger['benchmark'],'model':model,", "                common={'profiler_method_id':slots[-1]['profiler_method_id'],\n                        'profiler_version':slots[-1]['profiler_version'],\n                        'method_review_sha256':slots[-1]['method_review_sha256'],\n                        'benchmark':ledger['benchmark'],'model':model,")
sub(s,"                    points.append({'benchmark':ledger['benchmark'],'model':model,", "                    points.append({'profiler_method_id':slots[-1]['profiler_method_id'],\n                        'profiler_version':slots[-1]['profiler_version'],\n                        'method_review_sha256':slots[-1]['method_review_sha256'],\n                        'benchmark':ledger['benchmark'],'model':model,")
a=names[2]
sub(a,'import inspect\n','import inspect\nfrom v100_profiler_method import LEGACY_VERSION\n')
sub(a,"                if 'derived.json' in files:",'''                method_error = None
                if prov.get('status')=='not_started_profiler_method_preflight_hold':
                    require(sha(files['profiler_method_preflight.json'])==prov['method_preflight_sha256'],'Preflight hold SHA mismatch')
                    hold=json.loads(files['profiler_method_preflight.json'])
                    require(hold['status']=='rejected' and prov['new_gpu_processes']==prov['new_build_processes']==0 and prov['command']==[],'Invalid no-execution hold')
                    require(not any(n in files for n in ('counters.csv','derived.json','execution.log','build.log')),'No-execution hold contains target artifacts')
                    method_error='method preflight rejected; no target execution'
                elif prov.get('profiler_method') or (prov.get('profiler_version') and prov['profiler_version']!=LEGACY_VERSION):
                    try:
                        method_source=files['profiler_method_validator.py']
                        require(sha(method_source)==prov['profiler_method']['validator_sha256'],'Method validator SHA mismatch')
                        method_module=types.ModuleType('frozen_profiler_method')
                        exec(compile(method_source,'frozen_profiler_method','exec'),method_module.__dict__)
                        method_module.audit_profile(prov,files)
                        row['profiler_method']=prov['profiler_method']
                    except Exception as error:
                        method_error=repr(error)
                if method_error:
                    ledger=json.loads((study/'benchmarks'/(plan['benchmark']+'.json')).read_bytes())
                    require(identity not in ledger['models'][plan['model']].get('profile_ids',[]),'Unapproved method is selected')
                    qualifications.append({'profile_id':identity,'qualification':'Unapproved method retained, no positioning: '+method_error})
                    row['method_validation_status']='unapproved_retained_not_selected'
                if 'derived.json' in files and not method_error:''')
sub(a,"                    require(report['input_sha256'] == sha(files['counters.csv']), 'Derivation input SHA mismatch')", "                    require(report['input_sha256'] == sha(files['counters.csv']), 'Derivation input SHA mismatch')\n                    if prov.get('profiler_method'):\n                        require(report.get('profiler_method')==prov['profiler_method'],'Derived method mismatch')\n                        require(all(x.get('scope_context',{}).get('profiler_method')==prov['profiler_method'] for x in report['records']),'Kernel method mismatch')")
sub(a,"row['audit_status'] = ('pass_frozen_integrity_method_validation_hold' if row.get('allreduce_method_admission',{}).get('status')=='validation_hold' else 'pass')","row['audit_status'] = ('pass_frozen_integrity_method_validation_hold' if row.get('allreduce_method_admission',{}).get('status')=='validation_hold' or row.get('method_validation_status')=='unapproved_retained_not_selected' else 'pass')")
patch=[];bindings=[]
for n,text in modified.items():
 p=O/'proposed/tools'/n;p.write_text(text);patch.extend(difflib.unified_diff(original[n].splitlines(True),text.splitlines(True),fromfile='a/tools/'+n,tofile='b/tools/'+n));bindings.append({'file':'tools/'+n,'base_sha256':hashlib.sha256(original[n].encode()).hexdigest(),'proposed_sha256':hashlib.sha256(text.encode()).hexdigest()})
helper=O/'proposed/tools/v100_profiler_method.py';patch.extend(difflib.unified_diff([],helper.read_text().splitlines(True),fromfile='/dev/null',tofile='b/tools/v100_profiler_method.py'))
(O/'minimal.patch').write_text(''.join(patch));(O/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n');print(json.dumps(bindings,indent=2))
