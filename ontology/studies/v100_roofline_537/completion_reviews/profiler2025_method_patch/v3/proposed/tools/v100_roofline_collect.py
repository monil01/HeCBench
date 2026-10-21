#!/usr/bin/env python3
"""Collect audited, partial first-per-function/launch-shape V100 observations.

Consumes reviewed, freshly staged plans from v100_roofline_prepare.py. This is
profiling evidence, never a replacement for the accepted correctness/timing run.
No shell commands, privileged operations, or historical output directories.
"""
import argparse
import base64
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time
import types
import sys
import csv
import math

UUID = 'GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
METRICS = [
    'gpu__time_duration.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'dram__bytes.sum', 'dram__bytes.sum.peak_sustained',
    'dram__cycles_elapsed.avg.per_second', 'sm__cycles_elapsed.avg.per_second',
    'lts__t_bytes.sum', 'sm__inst_executed_pipe_tensor.sum',
    'lts__t_bytes.sum.peak_sustained', 'lts__cycles_elapsed.avg.per_second',
    'sm__inst_executed_pipe_tensor.sum.peak_sustained',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'smsp__sass_thread_inst_executed_op_integer_pred_on.sum',
] + ['smsp__sass_thread_inst_executed_op_' + x + '_pred_on.sum'
     for x in ('hadd', 'hmul', 'hfma', 'fadd', 'fmul', 'ffma', 'dadd', 'dmul', 'dfma')
] + ['sm__sass_thread_inst_executed_op_' + x + '_pred_on.sum.peak_sustained'
     for x in ('hfma', 'ffma', 'dfma')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def target_path_bindings(command, scratch, env):
    result={}
    for index,arg in enumerate(command):
        if arg.startswith('-'):
            continue
        path=Path(arg)
        if not path.is_absolute():
            resolved=shutil.which(arg,path=env.get('PATH')) if index==0 else None
            path=Path(resolved) if resolved else scratch/path
        try:
            if path.is_file() and (os.access(path,os.X_OK) or path.suffix in ('.jl','.py','.so')):
                result[str(path.resolve())]=sha(path)
        except (OSError,ValueError):
            # Inline script text and non-path CLI arguments are not filenames.
            continue
    return result


COLLECTOR_SOURCE = Path(__file__).read_bytes()
COLLECTOR_SHA256 = hashlib.sha256(COLLECTOR_SOURCE).hexdigest()


def supplemental_bindings(plan, root):
    """Bind current source context, without strengthening historical captures."""
    paths = set()
    extensions = {'.jl','.py','.cu','.cuh','.c','.cpp','.h','.hpp','.inc','.toml'}
    primary = root/'src'/(plan['benchmark']+'-julia')
    native = root/'src'/(plan['benchmark']+'-cuda')
    for directory in (primary, native):
        if directory.is_dir():
            paths.update(p for p in directory.rglob('*')
                         if p.is_file() and p.suffix in extensions)
    for asset in plan['runtime_assets']:
        paths.update(Path(x['path']) for x in asset['source_files'])
    review = (plan.get('review') or {}).get('entry', {})
    extra = review.get('additional_current_readonly_input_bindings', {})
    if isinstance(extra, dict):
        for path, expected in extra.items():
            if sha(path) != expected:
                raise RuntimeError('Reviewed additional input changed: '+path)
            paths.add(Path(path))
    elif isinstance(extra,list):
        for item in extra:
            if sha(item['path']) != item['sha256']:
                raise RuntimeError('Reviewed additional input changed: '+item['path'])
            paths.add(Path(item['path']))
    elif extra:
        raise RuntimeError('Additional reviewed inputs require explicit path/hash records')
    for arg in plan['profile_target_command']:
        if arg.startswith('--project='):
            project = Path(arg.split('=',1)[1])
            for name in ('Project.toml','Manifest.toml'):
                if (project/name).is_file():paths.add(project/name)
    return {str(p):sha(p) for p in sorted(paths)}


def check_supplemental(bound):
    for path, expected in bound.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('Current profile source/input context changed: '+path)


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.new')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bindings(plan):
    changed = [p for p, expected in plan['execution_bindings'].items()
               if not Path(p).is_file() or sha(p) != expected]
    if sha(plan['reference_result']) != plan['reference_result_sha256']:
        changed.append(plan['reference_result'])
    if changed:
        raise RuntimeError('Bound evidence changed: ' + repr(changed))


class GPUBusy(RuntimeError):
    pass


def gpu_processes():
    query = subprocess.run(['nvidia-smi','-i','6','--query-compute-apps=gpu_uuid,pid,process_name',
                            '--format=csv,noheader,nounits'], text=True,
                           capture_output=True, check=True)
    return [{'gpu_uuid':row[0].strip(),'pid':int(row[1].strip()),'name':Path(row[2].strip()).name}
            for row in csv.reader(query.stdout.splitlines()) if len(row)>=3 and row[0].strip()==UUID]


def idle_snapshot():
    processes=gpu_processes()
    if processes:
        raise GPUBusy('GPU6 has an existing compute process; defer execution: '+json.dumps(processes))
    device = subprocess.run(['nvidia-smi', '-i', '6',
        '--query-gpu=uuid,name,driver_version,memory.total,memory.used,clocks.current.sm,clocks.current.memory,power.draw,ecc.mode.current',
        '--format=csv'], text=True, capture_output=True, check=True).stdout
    if UUID not in device:
        raise RuntimeError('GPU6 UUID differs from protocol')
    return {'time': utc(), 'gpu': device, 'compute_processes': processes}


def wait_idle_snapshot(provenance,slot):
    waited=False
    while True:
        try:
            snapshot=idle_snapshot()
            if waited:
                provenance['idle_wait_finished_at']=utc()
                provenance['status']='started'
                dump(slot/'profile_provenance.json',provenance)
            return snapshot
        except GPUBusy as error:
            waited=True
            provenance.setdefault('idle_wait_started_at',utc())
            provenance.update(status='waiting_for_idle_gpu',idle_wait_last_checked_at=utc(),
                              idle_wait_reason=str(error))
            dump(slot/'profile_provenance.json',provenance)
            time.sleep(15)


def limits(max_file_bytes=1024**3):
    # New profiler auxiliary files are bounded. No accepted capture is touched.
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))


def own_session_members(session):
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid() or os.getsid(int(path.name))!=session:continue
            state=(path/'stat').read_text().split(')')[-1].split()[0]
            if state!='Z':members.append(int(path.name))
        except (OSError,ValueError):continue
    return members


def run(command, env, cwd, log, timeout, reserve, guard=None,monitor_gpu=False,artifact_policy=None):
    began = time.monotonic()
    started_at=utc()
    executed=list(command)
    if guard:
        executed=[sys.executable,guard['tool']]
        for directory in guard['write_directories']:
            executed+=['--allow-dir',directory]
        executed+=['--']+list(command)
    with log.open('wb') as out:
        process = subprocess.Popen(executed, cwd=cwd, env=env, stdout=out,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   preexec_fn=lambda:limits(artifact_policy['max_file_bytes'] if artifact_policy else 1024**3))
        stop_reason = None
        external_activity=[];last_gpu_check=0
        last_artifact_check=0
        while process.poll() is None:
            if time.monotonic() - began > timeout:
                stop_reason = 'wall_timeout'
            elif shutil.disk_usage(cwd).free < reserve:
                stop_reason = 'scratch_free_space_reserve'
            if artifact_policy and not stop_reason and time.monotonic()-last_artifact_check>=2:
                last_artifact_check=time.monotonic();payload=0;auxiliary=0
                for item in cwd.parent.rglob('*'):
                    if not item.is_file() or item.is_symlink():continue
                    size=item.stat().st_size;name=item.name
                    perlin=(artifact_policy['benchmark']=='perlin' and item.parent==cwd and name in ['perlin_trial'+str(i)+'.bin' for i in range(1,6)])
                    ztrial=(artifact_policy['benchmark']=='zmddft' and item.parent==cwd/'profile_outputs' and name in ['trial'+str(i)+'.bin' for i in range(1,6)])
                    zimpulse=(artifact_policy['benchmark']=='zmddft' and item.is_relative_to(cwd/'profile_outputs') and name==artifact_policy['model']+'_impulse_stages_f64.bin')
                    su3=(artifact_policy['benchmark']=='su3' and item.parent in (cwd,cwd/'profile_outputs') and name in ['su3_trial'+str(i)+'.bin' for i in range(1,6)])
                    if perlin or ztrial or zimpulse or su3:
                        payload+=size
                        cap=1541441088 if perlin else 1347849664 if su3 else 1073746496 if ztrial else 805306368
                        if size>cap:stop_reason='reviewed_payload_file_size_exceeded'
                    else:auxiliary+=size
                if payload>artifact_policy['payload_bytes']:stop_reason='reviewed_total_payload_exceeded'
                if auxiliary>artifact_policy['auxiliary_allowance_bytes']:stop_reason='reviewed_auxiliary_budget_exceeded'
            if monitor_gpu and not stop_reason and time.monotonic()-last_gpu_check>=2:
                last_gpu_check=time.monotonic()
                try:
                    observed=gpu_processes()
                except (OSError,ValueError,subprocess.SubprocessError) as error:
                    observed=[];stop_reason='GPU_activity_monitor_failed'
                    external_activity.append({'monitor_error':str(error),'observed_at':utc()})
                for item in observed:
                    owned=False;pid=item['pid']
                    try:
                        owned=os.getsid(pid)==process.pid
                        ancestor=pid
                        for _ in range(128):
                            if owned or ancestor==process.pid:
                                owned=True;break
                            if ancestor<=1:break
                            fields=(Path('/proc')/str(ancestor)/'stat').read_text().split(')')[-1].split()
                            ancestor=int(fields[1])
                    except (OSError,ValueError):pass
                    if not owned:external_activity.append({**item,'observed_at':utc()})
                if external_activity:stop_reason='external_gpu_activity_detected_own_profile_stopped'
            if stop_reason:
                # Only the fresh session created above; never discover/kill by name.
                if process.poll() is not None:break
                if os.getsid(process.pid) != process.pid:
                    raise RuntimeError('Own child session guard failed')
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.getsid(process.pid) != process.pid:
                        raise RuntimeError('Own child session guard failed')
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.5)
        code = process.wait()
        # NCU can leave a target/flusher alive after a timeout leader exits.
        # Do not hash/freeze counters or remove scratch until that own session
        # is quiescent. No other session, process name or user's work is signaled.
        quiescence_started=time.monotonic();descendant_cleanup=[]
        while True:
            members=own_session_members(process.pid)
            if not members:break
            if time.monotonic()-quiescence_started>10:
                if stop_reason is None:stop_reason='own_descendants_outlived_profile_leader'
                for pid in members:
                    try:
                        path=Path('/proc')/str(pid)
                        if path.stat().st_uid!=os.getuid() or os.getsid(pid)!=process.pid:
                            raise RuntimeError('Own session cleanup identity changed')
                        os.kill(pid,signal.SIGKILL);descendant_cleanup.append(pid)
                    except ProcessLookupError:pass
            time.sleep(.5)
    return {'returncode': code, 'stop_reason': stop_reason,
            'started_at':started_at,'finished_at':utc(),
            'external_gpu_activity':external_activity,
            'own_session_quiescence_seconds':time.monotonic()-quiescence_started,
            'own_descendants_cleaned':sorted(set(descendant_cleanup)),
            'wall_seconds': time.monotonic() - began, 'log_sha256': sha(log),
            'executed_command':executed}


def collect(plan_path, root, timeout, reserve_mib, retain_auxiliary, selection_path=None, tensor_modes=False, large_artifacts=False, application_replay_reviewed=False, profile_window_reviewed=False, isolated_assertion_reviewed=False, profiler_executable=None, profiler_method_review=None, profiler_method_review_sha256=None):
    # Child execution changes cwd to private scratch; repository/plan paths must
    # already be absolute before constructing the guarded launcher or evidence links.
    root = Path(root).resolve()
    plan_path = Path(plan_path).resolve()
    if selection_path is not None:
        selection_path = Path(selection_path).resolve()
    plan = json.loads(plan_path.read_text())
    active_gates=[g for g in plan['blocking_gates'] if not(isolated_assertion_reviewed and plan['benchmark']=='assert' and g=='requires_isolated_assertion_collector_phase_hooks')]
    reviewed_assert_ready=(isolated_assertion_reviewed and plan['benchmark']=='assert'
        and plan.get('staging_status')=='source_reviewed_waiting_explicit_isolated_assertion_collector_optin'
        and plan['blocking_gates']==['requires_isolated_assertion_collector_phase_hooks'])
    if active_gates or not(plan.get('staged') or reviewed_assert_ready):
        return {'plan': str(plan_path), 'status': 'not_executed_plan_blocked_or_not_staged'}
    slot = plan_path.parent
    if (slot / 'profile_provenance.json').exists():
        return {'plan': str(plan_path), 'status': 'existing_profile_preserved'}
    isolated_review = None
    if isolated_assertion_reviewed:
        if plan['benchmark']!='assert' or plan['model'] not in ('cuda_jl','ka','jacc'):
            raise RuntimeError('Isolated assertion phases restricted to reviewed Julia assert cases')
        link=plan['isolated_assertion_review'];review_path=Path(link['path'])
        if sha(review_path)!=link['sha256']:raise RuntimeError('Isolated assertion review changed')
        review_bytes=review_path.read_bytes();record=json.loads(review_bytes)['models'][link['entry']]
        if link['entry']!='assert/'+plan['model'] or record['configuration_id']!=plan['configuration_id'] or record['source_reviewed'] is not True:
            raise RuntimeError('Isolated assertion reviewed configuration mismatch')
        if record['reference_result_sha256']!=plan['reference_result_sha256'] or record['raw_config_sha256']!=plan['raw_config_sha256']:
            raise RuntimeError('Isolated assertion selected contract mismatch')
        if record['positive_command']!=plan['profile_target_command'] or any(record[k]!=plan[k] for k in ('pre_profile_check_command','post_profile_check_command')):
            raise RuntimeError('Isolated assertion phase commands mismatch')
        if (record['expected_negative_context_count'],record['expected_negative_status'],record['expected_positive_PERF_TRIAL_count'])!=(5,710,5):
            raise RuntimeError('Isolated assertion full check count changed')
        check_supplemental(record['execution_bindings'])
        if any(plan['execution_bindings'].get(name)!=digest for name,digest in record['execution_bindings'].items()):
            raise RuntimeError('Isolated assertion review not prospectively bound')
        (slot/'isolated_assertion_review.json').write_bytes(review_bytes)
        isolated_review={'review':link,'record':record,'runtime_validated_before_attempt':False}
    window_review = None
    if profile_window_reviewed:
        if plan['benchmark'] != 'rayleighBenardConvection':
            raise RuntimeError('Retained marker-window opt-in reviewed for RBC only')
        link=plan['profile_window_review'];review_path=Path(link['path'])
        if sha(review_path)!=link['sha256']:raise RuntimeError('Marker window review changed')
        review_bytes=review_path.read_bytes();record=json.loads(review_bytes)['models'][link['entry']]
        if link['entry']!=plan['benchmark']+'/'+plan['model'] or record['configuration_id']!=plan['configuration_id']:
            raise RuntimeError('Marker window reviewed configuration mismatch')
        if record['reference_result_sha256']!=plan['reference_result_sha256'] or record['raw_config_sha256']!=plan['raw_config_sha256']:
            raise RuntimeError('Marker window selected contract mismatch')
        if record['source_reviewed'] is not True or record['profile_from_start'] is not False or record['capture_all_launches_in_window'] is not False or record['replay_mode']!='kernel':
            raise RuntimeError('Unreviewed marker replay/coverage method')
        if (record['kernel_id'],record['filter_mode'],record['kernel_name_base'])!=('::regex:.*:1','per-launch-config','mangled'):
            raise RuntimeError('Unreviewed retained-window selector')
        if record['selected_window']['retained_trial']!=2 or record['selected_window']['zero_based_step']!=240 or record['source_expected_authored_dispatches']!=28:
            raise RuntimeError('Unreviewed scientific window')
        if record['original_GPU_function_bytes_sha256']!=record['copied_GPU_function_bytes_sha256'] or record['inverse_host_patch_recovers_original_header_exactly'] is not True:
            raise RuntimeError('Marker adapter changed GPU arithmetic')
        check_supplemental(record['execution_bindings'])
        if any(plan['execution_bindings'].get(name)!=digest for name,digest in record['execution_bindings'].items()):
            raise RuntimeError('Window source not prospectively bound by plan')
        (slot/'profile_window_review.json').write_bytes(review_bytes)
        window_review={'review':link,'record':record,'runtime_validated_before_attempt':False}
    application_review = None
    replay_mode = 'kernel'
    if application_replay_reviewed:
        if plan['benchmark'] != 'atomicSystemWide':
            raise RuntimeError('Application replay opt-in reviewed for atomicSystemWide only')
        link = plan['application_replay_review']
        review_path = Path(link['path'])
        if sha(review_path) != link['sha256']:
            raise RuntimeError('Application replay source review changed')
        review_bytes = review_path.read_bytes()
        record = json.loads(review_bytes)['models'][link['entry']]
        if link['entry'] != plan['benchmark']+'/'+plan['model'] or record['configuration_id'] != plan['configuration_id']:
            raise RuntimeError('Application replay reviewed configuration mismatch')
        if record['reference_result_sha256'] != plan['reference_result_sha256'] or record['raw_config_sha256'] != plan['raw_config_sha256']:
            raise RuntimeError('Application replay selected contract mismatch')
        if record['replay_mode'] != 'application' or record['app_replay_match'] != 'grid' or record['source_reviewed'] is not True:
            raise RuntimeError('Application replay method approval absent')
        check_supplemental(record['execution_bindings'])
        if any(plan['execution_bindings'].get(name) != digest for name,digest in record['execution_bindings'].items()):
            raise RuntimeError('Application replay review is not prospectively bound by the plan')
        (slot/'application_replay_review.json').write_bytes(review_bytes)
        application_review = {'review':link,'record':record,'runtime_validated_before_attempt':False}
        replay_mode = 'application'
    tensor_loader = None
    tensor_binding_inventory = {}
    if tensor_modes:
        if plan['benchmark'] != 'wmma':
            raise RuntimeError('Tensor mode opt-in is reviewed for WMMA only')
        loader_path = root/'tools/v100_roofline_tensor_modes.py'
        loader_source = loader_path.read_bytes()
        tensor_loader = types.ModuleType('bound_tensor_modes')
        tensor_loader.__file__ = str(loader_path)
        exec(compile(loader_source,str(loader_path),'exec'),tensor_loader.__dict__)
        tensor_binding_inventory = tensor_loader.required_bindings(plan['benchmark'],plan['model'])
        tensor_binding_inventory[str(loader_path)] = hashlib.sha256(loader_source).hexdigest()
        (slot/'tensor_mode_loader_source.py').write_bytes(loader_source)
    artifact_policy=None
    if large_artifacts:
        payloads={'perlin':7707205440,'zmddft':6174038848,'su3':6739248320}
        if plan['benchmark'] not in payloads:
            raise RuntimeError('Large-artifact opt-in reviewed for Perlin/ZMDDFT/SU3 only')
        artifact_policy={'benchmark':plan['benchmark'],'model':plan['model'],
            'max_file_bytes':2*1024**3,'payload_bytes':payloads[plan['benchmark']],
            'auxiliary_allowance_bytes':256*1024**2+plan.get('runtime_asset_bytes',0),
            'already_staged_runtime_asset_bytes':plan.get('runtime_asset_bytes',0),
            'scope':'Fresh per-process file limit and enforced per-attempt size budget; no OS/global limit changes. Full five outputs and independent checks retained.'}
    scratch = Path(plan['scratch_cwd']).resolve()
    scratch.relative_to(slot.resolve())
    scratch.relative_to(Path('/tmp'))
    if not scratch.is_dir() or scratch.stat().st_uid != os.getuid():
        raise RuntimeError('Scratch ownership/staging check failed')
    if plan['environment_overrides']['CUDA_VISIBLE_DEVICES'] != '6':
        raise RuntimeError('Only reviewed physical GPU6 is supported')
    env = os.environ.copy()
    env.update(plan['environment_overrides'])
    env['CUDA_VISIBLE_DEVICES'] = UUID
    for cache_name in ('cuda_cache','xdg_cache'):
        (scratch/cache_name).mkdir(exist_ok=True)
    env.update(CUDA_CACHE_PATH=str(scratch/'cuda_cache'),
               XDG_CACHE_HOME=str(scratch/'xdg_cache'),PYTHONDONTWRITEBYTECODE='1')
    cache_dirs=[Path('/home/imo/.julia')/x for x in ('compiled','logs','scratchspaces')]
    guard_path=root/'tools/v100_profile_write_guard.py'
    guard={'tool':str(guard_path),'sha256':sha(guard_path),
        'write_directories':[str(slot.resolve())]+[str(p.resolve())for p in cache_dirs if p.is_dir()],
        'policy':'Child-only Landlock writes restricted to fresh attempt and existing disposable Julia caches; existing character-device access retained. Reads retain ordinary user permissions. No unrestricted fallback.'}
    (slot/'write_guard_source.py').write_bytes(guard_path.read_bytes())
    # NCU's injected process needs Julia's bundled C++ runtime as well as libjulia.
    julia = Path('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu')
    env['LD_LIBRARY_PATH'] = str(julia/'lib/julia') + ':' + str(julia/'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    # Source-only method rejection is durable and uses zero target/GPU/build runs.
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
        return {'benchmark':plan['benchmark'],'model':plan['model'],'status':hold['status'],'error':repr(error)}
    if command[0] == 'julia':
        command[0] = '/home/imo/.juliaup/bin/julia'
    csv_path = slot / 'counters.csv'
    selection = ['--filter-mode','per-launch-config','--kernel-name-base','mangled',
                 '--kernel-id','::regex:.*:1']
    scope = 'first_invocation_per_mangled_specialization_and_launch_geometry; partial kernel sample'
    selection_override = None
    if selection_path:
        selected = json.loads(selection_path.read_text())
        if selected['benchmark'] == plan['benchmark']:
            if selected['configuration_id'] != plan['configuration_id']:
                raise RuntimeError('Selection override configuration mismatch')
            fixture = selected['fixture_description']
            if sha(fixture['path']) != fixture['sha256']:
                raise RuntimeError('Selection override fixture mismatch')
            selection = selected['selection'][plan['model']]
            scope = selected['coverage']
            selection_override = {'path':str(selection_path),'sha256':sha(selection_path),
                                  'expected_matching_launches':selected['expected_matching_launches']}
            if selected.get('source_recipe'):
                proof=selected['source_recipe']
                if sha(proof['path'])!=proof['sha256']:
                    raise RuntimeError('Retained-trial source recipe changed')
                row=next(x for x in selected['source_records'] if x['model']==plan['model'])
                if row['reference_result']['sha256']!=plan['reference_result_sha256'] or row['raw_config']['sha256']!=plan['raw_config_sha256']:
                    raise RuntimeError('Retained-trial selected contract mismatch')
                for binding in row['execution_bindings']+row.get('current_supplemental_sources',[]):
                    if sha(binding['path'])!=binding['sha256']:
                        raise RuntimeError('Retained-trial source binding changed')
                selection_override.update(retained_trial=selected['retained_trial'],source_recipe=proof,
                                          phase_proof='source_launch_ordinal_not_dynamic_range_trace')
    if window_review:
        if selection_path:raise RuntimeError('Marker window uses only its reviewed selector')
        selection+=['--profile-from-start','off']
        scope='Partial first-per-specialization/geometry observations inside source-checked retainedtrial2 step240; not complete timestep or1000step ROI'
    ncu = [ncu_path, '--target-processes', 'all'] + selection + [
           '--clock-control', 'none', '--cache-control', 'all', '--replay-mode',replay_mode,*(['--app-replay-match','grid','--app-replay-mode','strict'] if application_review else []),
           '--metrics', ','.join(METRICS), '--csv', '--page', 'raw',
           '--print-units', 'base', '--log-file', str(csv_path)] + command
    saved_result=json.loads(Path(plan['reference_result']).read_text())
    primary_mean=saved_result.get('models',{}).get(plan['model'],{}).get('mean_seconds')
    requested_timeout=timeout
    if isinstance(primary_mean,(int,float)) and math.isfinite(primary_mean) and primary_mean>0:
        timeout=max(timeout,min(3600,math.ceil(120+30*primary_mean)))
    provenance = {'schema_version': 1, 'started_at': utc(), 'plan': str(plan_path),
        'plan_sha256': sha(plan_path), 'collector_sha256': COLLECTOR_SHA256,
        'command': ncu, 'environment_overrides': {**plan['environment_overrides'],
            'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],
            'LD_LIBRARY_PATH': env['LD_LIBRARY_PATH'],
            'CUDA_CACHE_PATH':env['CUDA_CACHE_PATH'],'XDG_CACHE_HOME':env['XDG_CACHE_HOME'],
            'PYTHONDONTWRITEBYTECODE':env['PYTHONDONTWRITEBYTECODE']}, 'gpu_uuid': UUID,
        'selector_scope': scope, 'selection_override': selection_override,
        'ROI_coverage': 'unreviewed_partial; warmup/setup/validation may be present',
        'primary_contract_unchanged': True,
        'filesystem_write_guard':guard,
        'large_artifact_policy':artifact_policy,
        'operational_deadline':{'requested_seconds':requested_timeout,'effective_seconds':timeout,
            'saved_reported_mean_seconds':primary_mean,
            'rule':'max(requested,min(3600,ceil(120+30*saved_reported_mean))); coarse replay/process deadline, not a timing forecast. Reported time normalization may differ. Idle queue waiting is excluded.'},
        'profiling_auxiliary_retention': 'kept' if retain_auxiliary else
            'Fresh scratch only removed after manifest; not a correctness certificate. Raw counters, logs, provenance retained.',
        'status': 'started','replay_mode':replay_mode,'application_replay':application_review,'profile_window':window_review,'isolated_assertion_phases':isolated_review}
    if method:
        provenance.update(profiler_method=method_label,profiler_runtime_environment={k:env[k]for k in method['manifest']['pinned_runtime_environment']},profiler_version=version_text,profiler_executable=ncu_path,profiler_executable_sha256=sha(ncu_path))
    provenance['method_preflight_sha256']=sha(slot/'profiler_method_preflight.json')
    (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)
    dump(slot/'profile_provenance.json', provenance)
    try:
        if plan['benchmark']=='halo-finder' and env.get('HACC_NUM_CUDA_DEV') not in (None,'1'):
            raise RuntimeError('Reviewed serial Halo-finder requires HACC_NUM_CUDA_DEV unset or1')
        if plan['benchmark'] == 'streamUM':
            provenance['status'] = 'not_started_replay_method_blocked'
            raise RuntimeError('Kernel replay against concurrent CPU writes to other managed allocations and dynamic streams is not yet validated; retain source review and use a separately reviewed concurrency method.')
        if plan['benchmark'] == 'atomicSystemWide' and not application_review:
            provenance['status'] = 'not_started_replay_method_blocked'
            raise RuntimeError('Kernel replay cannot reproduce concurrent CPU/GPU atomics on the same managed allocation; requires separately reviewed application replay or single-pass diagnostics.')
        with Path('/tmp/hecbench-julia-study-gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            bindings(plan)
            if application_review:check_supplemental(application_review['record']['execution_bindings'])
            if window_review:check_supplemental(window_review['record']['execution_bindings'])
            current_bindings = supplemental_bindings(plan,root)
            for name, expected in tensor_binding_inventory.items():
                if sha(name) != expected:
                    raise RuntimeError('Tensor review dependency changed: '+name)
                current_bindings[name] = expected
            provenance['additional_current_profile_input_bindings'] = current_bindings
            if tensor_loader:
                provenance['tensor_mode_bindings_before_profile'] = tensor_binding_inventory
            provenance['additional_binding_scope'] = 'New profile only; never retroactive certification of historical correctness captures.'
            provenance['device_before'] = wait_idle_snapshot(provenance,slot)
            if artifact_policy:
                required_space = (artifact_policy['payload_bytes'] + artifact_policy['auxiliary_allowance_bytes']
                    - artifact_policy.get('already_staged_runtime_asset_bytes', 0) + reserve_mib * 1024**2)
                space_began = time.monotonic()
                space_waited = False
                while True:
                    free_space = shutil.disk_usage(scratch).free
                    if free_space >= required_space:
                        break
                    space_waited = True
                    elapsed = time.monotonic() - space_began
                    provenance['artifact_space_wait'] = {
                        'required_free_bytes': required_space, 'observed_free_bytes': free_space,
                        'last_checked_at': utc(), 'elapsed_seconds': elapsed, 'maximum_seconds': 300,
                        'scope': 'Prelaunch resource wait, outside profiling and original primary timing; no files removed.'}
                    provenance['status'] = 'waiting_for_reviewed_artifact_space'
                    dump(slot/'profile_provenance.json', provenance)
                    if elapsed >= 300:
                        raise RuntimeError('Free space cannot hold full reviewed five-trial payload, auxiliary allowance and reserve after bounded 300-second prelaunch wait')
                    time.sleep(2)
                if space_waited:
                    provenance['artifact_space_wait'].update(
                        satisfied_at=utc(), satisfied_free_bytes=free_space,
                        elapsed_seconds=time.monotonic()-space_began)
                    bindings(plan)
                    check_supplemental(current_bindings)
                    provenance['device_after_artifact_space_wait'] = wait_idle_snapshot(provenance, slot)
            elif shutil.disk_usage(scratch).free < reserve_mib * 1024**2:
                raise RuntimeError('Free space below reserve')
            if plan['build_command']:
                provenance['build'] = run(plan['build_command'], env, scratch,
                    slot/'build.log', timeout, reserve_mib*1024**2,guard)
                if provenance['build']['returncode'] != 0:
                    raise RuntimeError('Isolated build failed; see retained build log')
            if isolated_review:
                check_supplemental(isolated_review['record']['execution_bindings'])
                phase_env=env.copy();phase_env['HECBENCH_ASSERT_PHASE_AUTHORIZATION']=isolated_review['record']['authorization_marker']
                provenance['pre_profile_checks']=run(plan['pre_profile_check_command'],phase_env,scratch,
                    slot/'pre_profile_checks.log',timeout,reserve_mib*1024**2,guard,monitor_gpu=True)
                if provenance['pre_profile_checks']['returncode']!=0 or provenance['pre_profile_checks']['stop_reason'] is not None:
                    raise RuntimeError('Mandatory five isolated negative contexts failed; no positive NCU launch')
                bindings(plan);check_supplemental(isolated_review['record']['execution_bindings'])
            # Build/prechecks may create a GPU context; require a second idle check.
            provenance['device_immediately_before_profile'] = wait_idle_snapshot(provenance,slot)
            identity=subprocess.run(['nvidia-smi','-i','6','--query-gpu=uuid,pci.bus_id',
                '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
            expected_uuid,pci_text=next(csv.reader(identity.splitlines()))
            if expected_uuid.strip()!=UUID:raise RuntimeError('Expected UUID changed')
            domain,bus,device=pci_text.strip().split(':')
            expected_pci=(int(domain,16),int(bus,16),int(device.split('.')[0],16))
            provenance['expected_device_pci']=dict(domain=expected_pci[0],bus=expected_pci[1],device=expected_pci[2])
            provenance['profiler_version'] = version_text
            provenance['profiler_executable'] = ncu_path
            provenance['profiler_executable_sha256'] = sha(provenance['profiler_executable'])
            provenance['executed_target_path_bindings']=target_path_bindings(command,scratch,env)
            if method:
                method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)
                if {k:env[k]for k in method['manifest']['pinned_runtime_environment']}!=method['manifest']['pinned_runtime_environment']:raise RuntimeError('Pinned runtime drift')
            provenance['profile'] = run(ncu, env, scratch, slot/'execution.log',
                                        timeout, reserve_mib*1024**2,guard,monitor_gpu=True,artifact_policy=artifact_policy)
            bindings(plan)
            if method:method_gate.load(profiler_method_review,profiler_method_review_sha256,ncu_path,version_text)
            if isolated_review:
                check_supplemental(isolated_review['record']['execution_bindings'])
                outcome={'plan_sha256':sha(plan_path),'command':command,
                    'returncode':provenance['profile']['returncode'],'stop_reason':provenance['profile']['stop_reason'],
                    'own_session_quiescent':True,'guarded_under_GPU_lock':True,
                    'execution_log':{'path':str(slot/'execution.log'),'sha256':sha(slot/'execution.log'),'bytes':(slot/'execution.log').stat().st_size}}
                dump(slot/'positive_profile_outcome.json',outcome)
                provenance['post_profile_checks']=run(plan['post_profile_check_command'],phase_env,scratch,
                    slot/'post_profile_checks.log',timeout,reserve_mib*1024**2,guard)
                bindings(plan);check_supplemental(isolated_review['record']['execution_bindings'])
                if provenance['post_profile_checks']['returncode']!=0 or provenance['post_profile_checks']['stop_reason'] is not None:
                    raise RuntimeError('Mandatory joined assertion execution-only validation failed')
            if application_review:
                check_supplemental(application_review['record']['execution_bindings'])
                visible=[]
                for line in (slot/'execution.log').read_text(errors='replace').splitlines():
                    if line.startswith('STUDY_TRIAL '):visible.append(json.loads(line[len('STUDY_TRIAL '):]))
                groups=[visible[i:i+5] for i in range(0,len(visible),5)]
                passed=bool(groups) and all([r.get('trial') for r in g]==list(range(1,6)) and all(r.get('correct') is True and r.get('verification_contract')=='native_property_scores' for r in g) for g in groups)
                provenance['application_replay_visible_checks']={'five_trial_groups':len(groups),'all_visible_groups_pass':passed,'scope':'Visible full-program checks only; repeated application passes retain source fail-fast checks. No bitwise equality claim for nondeterministic exchange/CAS states.'}
                if provenance['profile']['returncode']==0 and provenance['profile']['stop_reason'] is None and not passed:
                    raise RuntimeError('Application replay visible full five-trial property checks absent or failed')
            if window_review:
                check_supplemental(window_review['record']['execution_bindings'])
                logs=(slot/'execution.log').read_text(errors='replace').splitlines()
                starts=[json.loads(l[len('RBC_PROFILE_WINDOW_BEGIN '):])for l in logs if l.startswith('RBC_PROFILE_WINDOW_BEGIN ')]
                ends=[json.loads(l[len('RBC_PROFILE_WINDOW_END '):])for l in logs if l.startswith('RBC_PROFILE_WINDOW_END ')]
                trials=[json.loads(l[len('STUDY_TRIAL '):])for l in logs if l.startswith('STUDY_TRIAL ')]
                passed=(len(starts)==len(ends)==1 and starts[0]['trial']==ends[0]['trial']==2 and starts[0]['zero_based_step']==ends[0]['zero_based_step']==240 and starts[0]['nu_samples_before']==1 and ends[0]['nu_samples_after']==2 and ends[0]['authored_dispatch_counts']==[3,6,2,11,5,0,1] and [t['trial']for t in trials]==list(range(1,6)) and all(t.get('correct') is True for t in trials))
                provenance['profile_window_runtime_checks']={'all_pass':passed,'begin':starts,'end':ends,'checked_trial_count':len(trials),'scope':'Exact marker/host-dispatch evidence; observed GPU kernel sample count separate, no complete-step claim.'}
                if provenance['profile']['returncode']==0 and provenance['profile']['stop_reason'] is None and not passed:
                    raise RuntimeError('Retained marker window or full five-trial checks failed')
            check_supplemental(current_bindings)
            check_supplemental(provenance['executed_target_path_bindings'])
            provenance['binding_recheck_after'] = 'unchanged'
            provenance['status'] = ('profile_process_completed' if provenance['profile']['returncode'] == 0 and provenance['profile']['stop_reason'] is None
                                    else 'profile_process_failed')
            if csv_path.is_file():
                provenance['counters_sha256'] = sha(csv_path)
                parser_path = root/'tools/v100_roofline_classify.py'
                parser_source = parser_path.read_bytes()
                (slot/'parser_source.py').write_bytes(parser_source)
                parser = types.ModuleType('roofline_parser')
                exec(compile(parser_source,str(parser_path),'exec'),parser.__dict__)
                try:
                    groups, csv_sha = parser.read_csv(csv_path)
                    if window_review and not groups:raise ValueError('Reviewed marker window produced no GPU samples; selector semantics unvalidated')
                    for group in groups:
                        actual_pci=tuple(group['metrics'].get('device__attribute_pci_'+key+'_id')
                                         for key in ('domain','bus','device'))
                        if actual_pci!=expected_pci:
                            raise ValueError('Profiler device PCI identity mismatch: '+repr(actual_pci))
                    provenance['profile_device_identity_verified']='All counter rows match NVML UUID-to-PCI mapping'
                    context = {k: plan[k] for k in ('benchmark','model','configuration_id','reference_result_sha256','raw_config_sha256')}
                    context.update(ROI_stage='unassigned', ROI_coverage='partial_first_function_geometry',
                                   first_invocation_may_be_warmup=True,
                                   cache_policy='all_cold_application_replay' if application_review else 'all_cold_replay',clock_control='none',replay_mode=replay_mode,
                                   selector_scope=scope,source_context_bound_before_after=True)
                    if window_review and provenance.get('profile_window_runtime_checks',{}).get('all_pass') and provenance['status']=='profile_process_completed':
                        context.update(retained_trial=2,zero_based_step=240,
                            profile_window_review=window_review['review'],
                            phase_proof='exact_source_markers_plus_runtime_begin_end_dispatch_and_five_check_receipts',
                            sample_phase='source_checked_trial2_step240_partial_specializations',
                            sample_in_primary_timed_invocation=True,first_invocation_may_be_warmup=False,
                            semantic_family_in_primary_ROI=True,ROI_stage='partial_samples_in_RK3_plus_Nu_timestep',
                            ROI_coverage='partial_first_specialization_geometry_in_retained_scientific_timestep')
                    if selection_override and selection_override.get('retained_trial'):
                        confirmed=len(groups)==selection_override['expected_matching_launches']
                        provenance['retained_window_expected_count_confirmed']=confirmed
                        context.update(retained_trial=selection_override['retained_trial'],
                            phase_proof=selection_override['phase_proof'],source_recipe=selection_override['source_recipe'],
                            sample_phase='source_counted_retained_trial2_scientific_window' if confirmed else 'retained_window_count_mismatch_phase_unproved',
                            sample_in_primary_timed_invocation=True if confirmed else None,
                            first_invocation_may_be_warmup=False if confirmed else True,
                            ROI_coverage='selected_scientific_family_window_not_elapsed_host_ROI' if confirmed else 'unproved_selector_count_mismatch')
                    calibration = root/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json'
                    proof=json.loads(calibration.read_text())
                    if proof['device_uuid']!=UUID or proof['architecture']!='sm_70':
                        raise ValueError('Packed-half calibration device mismatch')
                    provenance['operation_calibration']={'path':str(calibration.relative_to(root)),
                        'sha256':sha(calibration),'half_lane_factor':2,
                        'default_tensor_conversion_enabled':False,
                        'reason':'Each tensor conversion requires explicit per-row typed approval; default disabled. See tensor_mode_decisions when opt-in enabled.'}
                    if method:
                        prior_calibration=provenance['operation_calibration']
                        provenance['operation_calibration']={**method_label,'path':str(Path(profiler_method_review).resolve()),'sha256':method['review_sha256'],'half_lane_factor':2,'default_tensor_conversion_enabled':False,'prior_source_mode_calibration':prior_calibration}
                        context.update(profiler_method=method_label)
                    context.update(gpu_sm70=True,
                        fp16_work_semantics='Physical executed packed lanes, including unused scalarhalf high lanes; distinct from useful work.')
                    records = []
                    tensor_decisions = []
                    for group in groups:
                        row_context = dict(context)
                        tensor_args = {}
                        if tensor_loader:
                            decision = tensor_loader.load_tensor_mode(
                                benchmark=plan['benchmark'],model=plan['model'],
                                kernel_name=group['identity']['Kernel Name'],
                                kernel_name_base='demangled',context=row_context,
                                metrics=group['metrics'],captured_bindings=current_bindings)
                            tensor_decisions.append({'identity':group['identity'],'decision':decision})
                            if decision['approved']:
                                row_context.update(tensor_precision_mode=decision['tensor_precision_mode'],
                                    selected_target_SASS_observed=decision['selected_target_SASS_observed'],
                                    tensor_mode_basis='Source/API lowering proof, exact prospective dependency binding; no selected target SASS observation.')
                                tensor_args = {key:decision[key] for key in
                                    ('tensor_flops_per_instruction','tensor_convention_evidence')}
                        records.append(parser.derive(group,half_lanes=2,
                            half_provenance=provenance['operation_calibration'],
                            context=row_context,**tensor_args))
                    if tensor_loader:
                        dump(slot/'tensor_mode_decisions.json',{'schema_version':1,
                            'loader_sha256':sha(slot/'tensor_mode_loader_source.py'),
                            'profile_input_bindings':current_bindings,'records':tensor_decisions})
                        provenance['tensor_mode_approved_rows']=sum(x['decision']['approved'] for x in tensor_decisions)
                        provenance['tensor_mode_decisions_sha256']=sha(slot/'tensor_mode_decisions.json')
                    derived = {'schema_version': 1, 'input_csv': str(csv_path),
                        'input_sha256': csv_sha, 'parser_sha256': hashlib.sha256(parser_source).hexdigest(),
                        'kernel_count': len(records), 'records': records,
                        'benchmark_bottleneck': 'not_aggregated',
                        'raw_metrics': groups}
                    if method:derived['profiler_method']=method_label
                    dump(slot/'derived.json', derived)
                    provenance['kernel_count'] = len(records)
                    if selection_override:
                        provenance['selection_expected_launch_count_matches']=(len(records)==selection_override['expected_matching_launches'] if selection_override['expected_matching_launches'] is not None else None)
                except (ValueError, KeyError) as error:
                    provenance['counter_parse_error'] = str(error)
                    if provenance['status'] == 'profile_process_completed':
                        provenance['status'] = 'profile_completed_counter_derivation_failed'
    except Exception as error:
        if provenance['status'] != 'not_started_replay_method_blocked':
            provenance['status'] = 'collection_failed'
        provenance['error'] = str(error)
    finally:
        if artifact_policy:
            expected_files={}
            for trial in range(1,6):
                if plan['benchmark']=='perlin':expected_files[scratch/('perlin_trial'+str(trial)+'.bin')]=1541441088
                elif plan['benchmark']=='su3':
                    output=scratch if plan['model']=='cuda' else scratch/'profile_outputs'
                    expected_files[output/('su3_trial'+str(trial)+'.bin')]=1347849664
                else:expected_files[scratch/'profile_outputs'/('trial'+str(trial)+'.bin')]=1073746496
            if plan['benchmark']=='zmddft':
                expected_files[scratch/'profile_outputs/private_capture/diagnostics'/(plan['model']+'_impulse_stages_f64.bin')]=805306368
            validation=[{'path':str(path),'expected_bytes':expected,'actual_bytes':path.stat().st_size if path.is_file() else None}
                for path,expected in expected_files.items()]
            complete=all(row['actual_bytes']==row['expected_bytes'] for row in validation)
            provenance['large_artifact_validation']={'all_declared_payload_files_complete':complete,'files':validation,
                'scope':'Size completeness only, independent of numerical checks and rawcounter validity; auxiliary bytes are disposable only after hash manifest.'}
            if not complete and provenance['status']=='profile_process_completed':
                provenance['status']='profile_process_completed_artifact_incomplete'
        if isolated_review:
            retained=[]
            directory=slot/'negative_raw'
            for path in sorted(directory.iterdir()) if directory.is_dir() else []:
                if not path.is_file() or path.is_symlink():continue
                data=path.read_bytes();retained.append({'path':str(path.relative_to(slot)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64_bytes':base64.b64encode(data).decode('ascii')})
            dump(slot/'isolated_assertion_negative_raw.json',{'scope':'Exact fresh five-negative stdout/stderr/status logs, retained before any rejection; not numerical outputs or historical recovery.','files':retained})
        if application_review:
            passes=scratch/'profile_outputs/application_passes'
            retained=[]
            for path in sorted(passes.rglob('*')) if passes.is_dir() else []:
                if not path.is_file() or path.is_symlink():continue
                data=path.read_bytes()
                retained.append({'path':str(path.relative_to(scratch)), 'bytes':len(data),
                                 'sha256':hashlib.sha256(data).hexdigest(),
                                 'base64_bytes':base64.b64encode(data).decode('ascii')})
            dump(slot/'application_replay_pass_artifacts.json',{'scope':'Exact per-pass complete property artifacts, raw stdout and child status receipts; diagnostic/verification evidence separate from counter matching.', 'files':retained})
        for name in ('overlay_build_outcome.json','extra_build_outcome.json'):
            receipt=scratch/name
            if receipt.is_file():
                shutil.copy2(receipt,slot/'build_outcome.json')
        manifest = []
        for path in sorted(scratch.rglob('*')):
            if path.is_file() and not path.is_symlink():
                manifest.append({'path': str(path.relative_to(scratch)),
                                 'bytes': path.stat().st_size, 'sha256': sha(path)})
        dump(slot/'profiling_auxiliary_manifest.json', manifest)
        provenance['auxiliary_manifest_sha256'] = sha(slot/'profiling_auxiliary_manifest.json')
        if not retain_auxiliary:
            shutil.rmtree(scratch)
        provenance['finished_at'] = utc()
        dump(slot/'profile_provenance.json', provenance)
    # Root is the only measurement-field writer. Never call this from parallel GPU workers.
    ledger = root/'ontology/studies/v100_roofline_537/benchmarks'/ (plan['benchmark']+'.json')
    with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as metadata_lock:
        fcntl.flock(metadata_lock,fcntl.LOCK_EX)
        data = json.loads(ledger.read_text())
        model = data['models'][plan['model']]
        model.update(profiling_status=provenance['status'],
            profile_ids=[str(slot/'profile_provenance.json')],
            profile_provenance_sha256=sha(slot/'profile_provenance.json'),
            profile_plan_status=plan['staging_status'],
            profile_plan_blocking_gates=plan['blocking_gates'],
            profile_plan_configuration_id=plan['configuration_id'],
            sampled_kernel_count=provenance.get('kernel_count',0),
            roi_coverage='partial_first_specialization_geometry_unreviewed',
            roofline_position=None, roofline_region=None, observed_bottleneck=None,
            reason='Kernel observations require stage mapping; no whole-ROI bottleneck is inferred.')
        if (slot/'derived.json').exists():
            model['kernel_observations']={'path':str(slot/'derived.json'),'sha256':sha(slot/'derived.json')}
        dump(ledger,data)
    return {'benchmark':plan['benchmark'],'model':plan['model'],
            'status':provenance['status'],'kernels':provenance.get('kernel_count',0),
            'error':provenance.get('error')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--plan-root',type=Path,required=True)
    ap.add_argument('--exclude-benchmark',action='append',default=[])
    ap.add_argument('--benchmark',action='append',default=[])
    ap.add_argument('--skip-qualified-selected',action='store_true')
    ap.add_argument('--timeout',type=int,default=300)
    ap.add_argument('--reserve-mib',type=int,default=2048)
    ap.add_argument('--retain-auxiliary',action='store_true')
    ap.add_argument('--profiler-executable',type=Path)
    ap.add_argument('--profiler-method-review',type=Path)
    ap.add_argument('--profiler-method-review-sha256')
    ap.add_argument('--selection-overrides',type=Path)
    ap.add_argument('--tensor-modes',action='store_true',help='Explicit per-row source-bound WMMA mode approval')
    ap.add_argument('--large-artifacts',action='store_true',help='Reviewed Perlin/ZMDDFT/SU3 2GiB per-file policy, full payload space and auxiliary budget required')
    ap.add_argument('--application-replay-reviewed',action='store_true',help='Source-bound atomicSystemWide application/grid/strict replay only; retains all property checks')
    ap.add_argument('--profile-window-reviewed',action='store_true',help='Exact source/runtime-checked RBC retainedtrial2 step240 partial kernel samples only')
    ap.add_argument('--isolated-assertion-reviewed',action='store_true',help='Source-bound assert Julia full5negative/directpositive/postjoin phases, same guardedGPUlock')
    a=ap.parse_args()
    for p in sorted(a.plan_root.glob('*/*/plan.json')):
        plan=json.loads(p.read_text())
        if plan['benchmark'] in a.exclude_benchmark:continue
        if a.benchmark and plan['benchmark'] not in a.benchmark:continue
        if a.skip_qualified_selected:
            ledger=json.loads((a.source_root/'ontology/studies/v100_roofline_537/benchmarks'/(plan['benchmark']+'.json')).read_text())
            selected=ledger['models'][plan['model']]
            if selected['profiling_status']=='profile_process_completed' and selected.get('sampled_kernel_count',0)>0:
                print(json.dumps({'benchmark':plan['benchmark'],'model':plan['model'],'status':'already_qualified_selected_capture_preserved'}),flush=True)
                continue
        result=collect(p,a.source_root,a.timeout,a.reserve_mib,a.retain_auxiliary,a.selection_overrides,a.tensor_modes,a.large_artifacts,a.application_replay_reviewed,a.profile_window_reviewed,a.isolated_assertion_reviewed,a.profiler_executable,a.profiler_method_review,a.profiler_method_review_sha256)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
