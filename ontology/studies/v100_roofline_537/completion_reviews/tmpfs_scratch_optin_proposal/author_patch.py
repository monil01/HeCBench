# CPU text authoring only: never imports or executes either proposed module.
from pathlib import Path
import difflib,hashlib,json
base=Path('/home/imo/HeCBench_julia/tools/v100_roofline_collect.py')
out=Path('/tmp/roofline-completion-20261021/tmpfs_scratch_optin_proposal')
source=base.read_text(); expected='c21b1edd4c6e0924250b306e1b97ff7ff1371b8407eb92899e089dbdca364116'
assert hashlib.sha256(base.read_bytes()).hexdigest()==expected
(out/'collector_C21_original.py').write_bytes(base.read_bytes())
new=source

def replace(old,replacement):
    global new
    assert new.count(old)==1,(old,new.count(old))
    new=new.replace(old,replacement)

replace('def run(command, env, cwd, log, timeout, reserve, guard=None,monitor_gpu=False,artifact_policy=None):',
        'def run(command, env, cwd, log, timeout, reserve, guard=None,monitor_gpu=False,artifact_policy=None,storage_policy=None):')
replace("    began = time.monotonic()\n    started_at=utc()", "    if storage_policy: storage_policy.verify_bindings(); storage_policy.sample()\n    began = time.monotonic()\n    started_at=utc()")
replace("            if artifact_policy and not stop_reason and time.monotonic()-last_artifact_check>=2:",
"""            if storage_policy and not stop_reason:
                try:
                    storage_policy.sample()
                except Exception as storage_error:
                    stop_reason='reviewed_tmpfs_storage_guard_failed'
                    storage_policy.last_failure=repr(storage_error)
            if artifact_policy and not stop_reason and time.monotonic()-last_artifact_check>=2:""")
replace("    return {'returncode': code, 'stop_reason': stop_reason,", """    if storage_policy:
        try:
            storage_policy.sample()
            storage_policy.verify_bindings()
        except Exception as storage_error:
            stop_reason=stop_reason or 'reviewed_tmpfs_storage_guard_failed'
            storage_policy.last_failure=repr(storage_error)
    result = {'returncode': code, 'stop_reason': stop_reason,""")
replace("            'executed_command':executed}\n\n\ndef collect", """            'executed_command':executed}
    if storage_policy:
        result['reviewed_tmpfs_storage']=storage_policy.summary()
        result['reviewed_tmpfs_storage']['last_failure']=getattr(storage_policy,'last_failure',None)
    return result


def collect""")
replace('profiler_method_review_sha256=None):\n    # Child execution',
        'profiler_method_review_sha256=None,tmpfs_scratch_review=None,tmpfs_scratch_review_sha256=None):\n    # Child execution')
replace("    scratch = Path(plan['scratch_cwd']).resolve()\n    scratch.relative_to(slot.resolve())\n    scratch.relative_to(Path('/tmp'))",
"""    scratch = Path(plan['scratch_cwd']).resolve()
    storage_policy=None
    storage_source=None
    if tmpfs_scratch_review or tmpfs_scratch_review_sha256:
        if not(tmpfs_scratch_review and tmpfs_scratch_review_sha256):
            raise RuntimeError('Tmpfs opt-in requires exact review path AND SHA')
        # Import only this opt-in branch; legacy branch remains unchanged.
        import v100_tmpfs_scratch as storage_gate
        storage_source=Path(storage_gate.__file__).read_bytes()
        storage_policy=storage_gate.Policy(tmpfs_scratch_review,tmpfs_scratch_review_sha256,
            plan_path,plan,slot,COLLECTOR_SHA256,hashlib.sha256(storage_source).hexdigest(),
            sha(root/'tools/v100_profile_write_guard.py'),retain_auxiliary,artifact_policy,
            reserve_mib*1024**2)
        (slot/'tmpfs_storage_policy_source.py').write_bytes(storage_source)
        (slot/'tmpfs_storage_review.json').write_bytes(storage_policy.review_bytes)
    else:
        scratch.relative_to(slot.resolve())
        scratch.relative_to(Path('/tmp'))""")
replace("    (slot/'write_guard_source.py').write_bytes(guard_path.read_bytes())",
"""    if storage_policy:
        guard['write_directories'].append(str(scratch))
        guard['reviewed_tmpfs_scratch']={'review_sha256':storage_policy.review_sha,
            'scope':'Only exact private scratch leaf added; no /dev/shm-wide permission.'}
    (slot/'write_guard_source.py').write_bytes(guard_path.read_bytes())""")
replace("            60,reserve_mib*1024**2,guard,monitor_gpu=False)",
        "            60,reserve_mib*1024**2,guard,monitor_gpu=False,storage_policy=storage_policy)")
replace("        (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)\n        dump(slot/'profile_provenance.json',hold)",
"""        if storage_policy:
            hold['reviewed_tmpfs_storage']=storage_policy.summary()
            if getattr(storage_policy,'last_failure',None):
                hold['status']='not_started_tmpfs_storage_hold'
        (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)
        dump(slot/'profile_provenance.json',hold)""")
replace("    if method:\n        provenance.update(profiler_method=method_label,", "    if storage_policy:\n        provenance['reviewed_tmpfs_storage']=storage_policy.summary()\n    if method:\n        provenance.update(profiler_method=method_label,")
replace("            provenance['device_before'] = wait_idle_snapshot(provenance,slot)",
"""            provenance['device_before'] = wait_idle_snapshot(provenance,slot)
            if storage_policy: storage_policy.sample()""")
replace("                    slot/'build.log', timeout, reserve_mib*1024**2,guard)",
        "                    slot/'build.log', timeout, reserve_mib*1024**2,guard,storage_policy=storage_policy)")
replace("                if provenance['build']['returncode'] != 0:",
        "                if provenance['build']['returncode'] != 0 or (storage_policy and provenance['build']['stop_reason'] is not None):")
replace("                    slot/'pre_profile_checks.log',timeout,reserve_mib*1024**2,guard,monitor_gpu=True)",
        "                    slot/'pre_profile_checks.log',timeout,reserve_mib*1024**2,guard,monitor_gpu=True,storage_policy=storage_policy)")
replace("                                        timeout, reserve_mib*1024**2,guard,monitor_gpu=True,artifact_policy=artifact_policy)",
        "                                        timeout, reserve_mib*1024**2,guard,monitor_gpu=True,artifact_policy=artifact_policy,storage_policy=storage_policy)")
replace("                    slot/'post_profile_checks.log',timeout,reserve_mib*1024**2,guard)",
        "                    slot/'post_profile_checks.log',timeout,reserve_mib*1024**2,guard,storage_policy=storage_policy)")
replace("        if not retain_auxiliary:\n            shutil.rmtree(scratch)",
"""        if storage_policy:
            # Always retain full five payloads, even on failed/partial runs.
            # Policy construction requires --retain-auxiliary; no tmpfs cleanup here.
            provenance['reviewed_tmpfs_storage']=storage_policy.summary()
            provenance['reviewed_tmpfs_storage']['manifest_sha256']=provenance['auxiliary_manifest_sha256']
        if not retain_auxiliary:
            shutil.rmtree(scratch)""")
replace("    ap.add_argument('--profiler-method-review-sha256')", """    ap.add_argument('--profiler-method-review-sha256')
    ap.add_argument('--tmpfs-scratch-review',type=Path)
    ap.add_argument('--tmpfs-scratch-review-sha256')""")
replace('a.profiler_executable,a.profiler_method_review,a.profiler_method_review_sha256)',
        'a.profiler_executable,a.profiler_method_review,a.profiler_method_review_sha256,a.tmpfs_scratch_review,a.tmpfs_scratch_review_sha256)')
(out/'proposed/tools/v100_roofline_collect.py').write_text(new)
(out/'minimal.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='a/tools/v100_roofline_collect.py',tofile='b/tools/v100_roofline_collect.py')))
print(json.dumps({'base_sha256':expected,'candidate_sha256':hashlib.sha256(new.encode()).hexdigest(),'diff_lines':len((out/'minimal.patch').read_text().splitlines()),'executed_proposed_modules':False}))
