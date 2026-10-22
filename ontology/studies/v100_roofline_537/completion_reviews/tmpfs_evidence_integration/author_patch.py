from pathlib import Path
import hashlib,difflib,json
O=Path(__file__).parent;P=O/'proposed/tools';B=O/'bases';H=lambda b:hashlib.sha256(b).hexdigest()
p=P/'v100_tmpfs_evidence.py';s=p.read_text();s=s if 'retained_preflight_hold' in s else s.replace("    require(set(inventory['snapshots'])>=", "    if prov['status']=='not_started_tmpfs_storage_preflight_hold':\n        return {'status':'retained_preflight_hold','positions_publishable':False,'source':'offline frozen available bytes only; missing inputs remain explicit'}\n    require(set(inventory['snapshots'])>=");p.write_text(s)
s=(B/'v100_roofline_collect_tmpfs.py').read_text();s=s.replace('    storage_source=None\n','    storage_source=None\n    storage_evidence=None\n',1)
a=s.index('        storage_policy=storage_gate.Policy(');z=s.index('\n    else:\n        scratch.relative_to',a)
s=s[:a]+'''        # Snapshot only tiny source/plan bytes; no payload relocation or SDK reads.
        snapshots={}
        validator_path=Path(__file__).with_name('v100_tmpfs_evidence.py')
        validator_bytes=validator_path.read_bytes()
        def snapshot(name,path,expected=None):
            data=Path(path).read_bytes()
            if expected is not None and hashlib.sha256(data).hexdigest()!=expected:
                raise RuntimeError('Tmpfs snapshot SHA mismatch: '+name)
            (slot/name).write_bytes(data)
            snapshots[name]={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        def snapshot_manifest():
            data={'schema_version':1,'scope':'Storage review/source evidence only; original plan and tiny native source retained, no new numerical verdict.','snapshots':snapshots}
            dump(slot/'tmpfs_storage_evidence.json',data)
            return {'manifest_sha256':sha(slot/'tmpfs_storage_evidence.json'),
                    'validator_sha256':hashlib.sha256(validator_bytes).hexdigest()}
        try:
            snapshot('tmpfs_evidence_validator.py',validator_path)
            (slot/'tmpfs_storage_policy_source.py').write_bytes(storage_source)
            snapshots['tmpfs_storage_policy_source.py']={'path':storage_gate.__file__,
                'sha256':hashlib.sha256(storage_source).hexdigest(),'bytes':len(storage_source)}
            snapshot('tmpfs_storage_review.json',tmpfs_scratch_review,tmpfs_scratch_review_sha256)
            storage_review=json.loads((slot/'tmpfs_storage_review.json').read_bytes())
            if storage_review['evidence_validator_sha256']!=hashlib.sha256(validator_bytes).hexdigest():
                raise RuntimeError('Tmpfs evidence validator not approved')
            row=storage_review['models'][plan['benchmark']+'/'+plan['model']]
            snapshot('tmpfs_original_plan.json',row['original_plan_path'],row['original_plan_sha256'])
            if plan['model']=='cuda':
                proof=row['native_failure_diagnostic_overlay']
                snapshot('tmpfs_native_original.cu',proof['original_source_archive'],proof['original_source_sha256'])
                snapshot('tmpfs_native_overlay.cu',proof['fresh_source_path'],proof['fresh_source_sha256'])
            storage_evidence=snapshot_manifest()
            storage_policy=storage_gate.Policy(tmpfs_scratch_review,tmpfs_scratch_review_sha256,
                plan_path,plan,slot,COLLECTOR_SHA256,hashlib.sha256(storage_source).hexdigest(),
                sha(root/'tools/v100_profile_write_guard.py'),retain_auxiliary,artifact_policy,
                reserve_mib*1024**2)
        except Exception as storage_error:
            # Retain rejected available inputs too; no target/build/profile began.
            storage_evidence=snapshot_manifest()
            (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)
            (slot/'write_guard_source.py').write_bytes((root/'tools/v100_profile_write_guard.py').read_bytes())
            hold={'benchmark':plan['benchmark'],'model':plan['model'],'gpu_uuid':UUID,
                'plan_sha256':sha(plan_path),'collector_sha256':COLLECTOR_SHA256,
                'status':'not_started_tmpfs_storage_preflight_hold','finished_at':utc(),
                'selector_scope':'No target/build/profile launch; tmpfs preflight rejected',
                'command':[],'error':repr(storage_error),'new_gpu_processes':0,'new_build_processes':0,
                'reviewed_tmpfs_storage':{'review_sha256':tmpfs_scratch_review_sha256},
                'tmpfs_storage_evidence':storage_evidence}
            dump(slot/'profile_provenance.json',hold)
            return {'benchmark':plan['benchmark'],'model':plan['model'],'status':hold['status'],'error':repr(storage_error)}
'''+s[z:]
s=s.replace("hold['reviewed_tmpfs_storage']=storage_policy.summary()","hold['reviewed_tmpfs_storage']=storage_policy.summary()\n            hold['tmpfs_storage_evidence']=storage_evidence")
s=s.replace("provenance['reviewed_tmpfs_storage']=storage_policy.summary()","provenance['reviewed_tmpfs_storage']=storage_policy.summary()\n        provenance['tmpfs_storage_evidence']=storage_evidence",1)
(P/'v100_roofline_collect_tmpfs.py').write_text(s)
# Frozen helper is loaded only in explicit tmpfs branches; no ordinary-path import.
s=(B/'v100_roofline_summarize.py').read_text();s=s.replace("'allreduce_execution_authorization.json'):","'allreduce_execution_authorization.json',\n                 'tmpfs_storage_policy_source.py','tmpfs_storage_review.json','tmpfs_original_plan.json',\n                 'tmpfs_native_original.cu','tmpfs_native_overlay.cu','tmpfs_evidence_validator.py','tmpfs_storage_evidence.json'):")
a=s.index('        runtime_review=applicability.get(');s=s[:a]+'''        storage_admission=None;storage_error=None
        if prov.get('reviewed_tmpfs_storage') or prov.get('tmpfs_storage_evidence'):
            try:
                import v100_tmpfs_evidence as storage_gate
                available={n:(slot/n).read_bytes()for n in (*storage_gate.NAMES,'plan.json','collector_source.py','write_guard_source.py')if(slot/n).is_file()}
                if available.get('tmpfs_evidence_validator.py')!=Path(storage_gate.__file__).read_bytes():
                    raise ValueError('Unreviewed tmpfs evidence-validator bytes')
                storage_admission=storage_gate.validate(prov,available)
                if not storage_admission or storage_admission['status']!='validated':
                    raise ValueError('Retained tmpfs preflight hold; no target admission')
            except Exception as error:
                storage_error=repr(error)
        '''+s[a:].lstrip()
s=s.replace("        if method_error:\n            attempt.update", "        if storage_error:\n            attempt.update(status='tmpfs_storage_evidence_validation_hold',storage_validation_error=storage_error)\n        if storage_admission is not None:attempt['tmpfs_storage_admission']=storage_admission\n        if method_error:\n            attempt.update",1)
a=s.index('        if method_error:\n            for saved_attempt');s=s[:a]+'''        if storage_error:
            # Retain failed attempt evidence; do not poison another selected capture.
            for saved_attempt in attempts:
                if saved_attempt['profile_id']==attempt['profile_id']:saved_attempt.update(attempt)
            if str(target.relative_to(root))in state.get('profile_ids',[]):
                state.update(profiling_status='tmpfs_storage_evidence_validation_hold',profile_ids=[],sampled_kernel_count=0,sampled_kernel_precision_regimes={},sampled_kernel_predicted_resource_limits={},roi_coverage='tmpfs_evidence_hold_no_positioning',roofline_position=None,roofline_region=None,observed_bottleneck=None)
                state.pop('kernel_observations',None)
            dump(path,ledger)
            continue
'''+s[a:]
(P/'v100_roofline_summarize.py').write_text(s)
s=(B/'v100_roofline_audit.py').read_text();a=s.index("                if prov.get('profile_window'):");s=s[:a]+'''                if prov.get('reviewed_tmpfs_storage') or prov.get('tmpfs_storage_evidence'):
                    # Use exact frozen validator; never consult volatile scratch/SDK.
                    source=files['tmpfs_evidence_validator.py']
                    require(sha(source)==prov['tmpfs_storage_evidence']['validator_sha256'],'Tmpfs frozen validator SHA mismatch')
                    gate=types.ModuleType('frozen_tmpfs_evidence')
                    exec(compile(source,'<frozen_tmpfs_evidence>','exec'),gate.__dict__)
                    admission=gate.validate(prov,files)
                    require(admission is not None,'Tmpfs admission absent')
                    row['tmpfs_storage_admission']=admission
                    if admission['status']!='validated':
                        require(prov['status']=='not_started_tmpfs_storage_preflight_hold','Unapproved tmpfs attempt')
                        qualifications.append({'profile_id':identity,'reason':'Retained tmpfs preflight rejection; no GPU/build/profile attempt admitted'})
'''+s[a:];(P/'v100_roofline_audit.py').write_text(s)
# Diff only; source generation is not an import, test or execution of candidates.
for name in ['v100_roofline_collect_tmpfs.py','v100_roofline_summarize.py','v100_roofline_audit.py']:
 (O/(name+'.patch')).write_text(''.join(difflib.unified_diff((B/name).read_text().splitlines(True),(P/name).read_text().splitlines(True),fromfile='base/'+name,tofile='proposed/'+name)))
