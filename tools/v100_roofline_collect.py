#!/usr/bin/env python3
"""Collect audited, partial first-per-function/launch-shape V100 observations.

Consumes reviewed, freshly staged plans from v100_roofline_prepare.py. This is
profiling evidence, never a replacement for the accepted correctness/timing run.
No shell commands, privileged operations, or historical output directories.
"""
import argparse
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


def idle_snapshot():
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                            '--format=csv,noheader,nounits'], text=True,
                           capture_output=True, check=True)
    processes = query.stdout
    if any(line.split(',')[0].strip() == UUID for line in processes.splitlines()):
        raise RuntimeError('GPU6 has an existing compute process; refuse execution: ' + processes)
    device = subprocess.run(['nvidia-smi', '-i', '6',
        '--query-gpu=uuid,name,driver_version,memory.total,memory.used,clocks.current.sm,clocks.current.memory,power.draw,ecc.mode.current',
        '--format=csv'], text=True, capture_output=True, check=True).stdout
    if UUID not in device:
        raise RuntimeError('GPU6 UUID differs from protocol')
    return {'time': utc(), 'gpu': device, 'compute_processes': processes}


def limits():
    # New profiler auxiliary files are bounded. No accepted capture is touched.
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**3, 1024**3))


def run(command, env, cwd, log, timeout, reserve, guard=None):
    began = time.monotonic()
    executed=list(command)
    if guard:
        executed=[sys.executable,guard['tool']]
        for directory in guard['write_directories']:
            executed+=['--allow-dir',directory]
        executed+=['--']+list(command)
    with log.open('wb') as out:
        process = subprocess.Popen(executed, cwd=cwd, env=env, stdout=out,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   preexec_fn=limits)
        stop_reason = None
        while process.poll() is None:
            if time.monotonic() - began > timeout:
                stop_reason = 'wall_timeout'
            elif shutil.disk_usage(cwd).free < reserve:
                stop_reason = 'scratch_free_space_reserve'
            if stop_reason:
                # Only the fresh session created above; never discover/kill by name.
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
    return {'returncode': code, 'stop_reason': stop_reason,
            'wall_seconds': time.monotonic() - began, 'log_sha256': sha(log),
            'executed_command':executed}


def collect(plan_path, root, timeout, reserve_mib, retain_auxiliary, selection_path=None):
    plan = json.loads(plan_path.read_text())
    if plan['blocking_gates'] or not plan.get('staged'):
        return {'plan': str(plan_path), 'status': 'not_executed_plan_blocked_or_not_staged'}
    slot = plan_path.parent
    if (slot / 'profile_provenance.json').exists():
        return {'plan': str(plan_path), 'status': 'existing_profile_preserved'}
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
    command = list(plan['profile_target_command'])
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
    ncu = ['ncu', '--target-processes', 'all'] + selection + [
           '--clock-control', 'none', '--cache-control', 'all', '--replay-mode','kernel',
           '--metrics', ','.join(METRICS), '--csv', '--page', 'raw',
           '--print-units', 'base', '--log-file', str(csv_path)] + command
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
        'profiling_auxiliary_retention': 'kept' if retain_auxiliary else
            'Fresh scratch only removed after manifest; not a correctness certificate. Raw counters, logs, provenance retained.',
        'status': 'started'}
    (slot/'collector_source.py').write_bytes(COLLECTOR_SOURCE)
    dump(slot/'profile_provenance.json', provenance)
    try:
        if plan['benchmark'] == 'atomicSystemWide':
            provenance['status'] = 'not_started_replay_method_blocked'
            raise RuntimeError('Kernel replay cannot reproduce concurrent CPU/GPU atomics on the same managed allocation; requires separately reviewed application replay or single-pass diagnostics.')
        with Path('/tmp/hecbench-julia-study-gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            bindings(plan)
            current_bindings = supplemental_bindings(plan,root)
            provenance['additional_current_profile_input_bindings'] = current_bindings
            provenance['additional_binding_scope'] = 'New profile only; never retroactive certification of historical correctness captures.'
            provenance['device_before'] = idle_snapshot()
            if shutil.disk_usage(scratch).free < reserve_mib * 1024**2:
                raise RuntimeError('Free space below reserve')
            if plan['build_command']:
                provenance['build'] = run(plan['build_command'], env, scratch,
                    slot/'build.log', timeout, reserve_mib*1024**2,guard)
                if provenance['build']['returncode'] != 0:
                    raise RuntimeError('Isolated build failed; see retained build log')
            # Build may create a GPU context; require a second idle check.
            provenance['device_immediately_before_profile'] = idle_snapshot()
            identity=subprocess.run(['nvidia-smi','-i','6','--query-gpu=uuid,pci.bus_id',
                '--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
            expected_uuid,pci_text=next(csv.reader(identity.splitlines()))
            if expected_uuid.strip()!=UUID:raise RuntimeError('Expected UUID changed')
            domain,bus,device=pci_text.strip().split(':')
            expected_pci=(int(domain,16),int(bus,16),int(device.split('.')[0],16))
            provenance['expected_device_pci']=dict(domain=expected_pci[0],bus=expected_pci[1],device=expected_pci[2])
            provenance['profiler_version'] = subprocess.run(['ncu', '--version'],
                capture_output=True, text=True, check=True).stdout
            provenance['profile'] = run(ncu, env, scratch, slot/'execution.log',
                                        timeout, reserve_mib*1024**2,guard)
            bindings(plan)
            check_supplemental(current_bindings)
            provenance['binding_recheck_after'] = 'unchanged'
            provenance['status'] = ('profile_process_completed' if provenance['profile']['returncode'] == 0
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
                    for group in groups:
                        actual_pci=tuple(group['metrics'].get('device__attribute_pci_'+key+'_id')
                                         for key in ('domain','bus','device'))
                        if actual_pci!=expected_pci:
                            raise ValueError('Profiler device PCI identity mismatch: '+repr(actual_pci))
                    provenance['profile_device_identity_verified']='All counter rows match NVML UUID-to-PCI mapping'
                    context = {k: plan[k] for k in ('benchmark','model','configuration_id','reference_result_sha256')}
                    context.update(ROI_stage='unassigned', ROI_coverage='partial_first_function_geometry',
                                   first_invocation_may_be_warmup=True,
                                   cache_policy='all_cold_replay',clock_control='none',
                                   selector_scope=scope,source_context_bound_before_after=True)
                    calibration = root/'ontology/studies/v100_roofline_537/calibration/v100_sm70_20261020/review.json'
                    proof=json.loads(calibration.read_text())
                    if proof['device_uuid']!=UUID or proof['architecture']!='sm_70':
                        raise ValueError('Packed-half calibration device mismatch')
                    provenance['operation_calibration']={'path':str(calibration.relative_to(root)),
                        'sha256':sha(calibration),'half_lane_factor':2,
                        'tensor_conversion_enabled':False,
                        'reason':'Tensor target arithmetic mode requires per-kernel source/SASS review; presence remains recorded.'}
                    context.update(gpu_sm70=True,
                        fp16_work_semantics='Physical executed packed lanes, including unused scalarhalf high lanes; distinct from useful work.')
                    records = [parser.derive(g,half_lanes=2,
                        half_provenance=provenance['operation_calibration'],context=context)for g in groups]
                    derived = {'schema_version': 1, 'input_csv': str(csv_path),
                        'input_sha256': csv_sha, 'parser_sha256': hashlib.sha256(parser_source).hexdigest(),
                        'kernel_count': len(records), 'records': records,
                        'benchmark_bottleneck': 'not_aggregated',
                        'raw_metrics': groups}
                    dump(slot/'derived.json', derived)
                    provenance['kernel_count'] = len(records)
                    if selection_override:
                        provenance['selection_expected_launch_count_matches']=(len(records)==selection_override['expected_matching_launches'])
                except (ValueError, KeyError) as error:
                    provenance['counter_parse_error'] = str(error)
    except Exception as error:
        if provenance['status'] != 'not_started_replay_method_blocked':
            provenance['status'] = 'collection_failed'
        provenance['error'] = str(error)
    finally:
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
    ap.add_argument('--timeout',type=int,default=300)
    ap.add_argument('--reserve-mib',type=int,default=2048)
    ap.add_argument('--retain-auxiliary',action='store_true')
    ap.add_argument('--selection-overrides',type=Path)
    a=ap.parse_args()
    for p in sorted(a.plan_root.glob('*/*/plan.json')):
        result=collect(p,a.source_root,a.timeout,a.reserve_mib,a.retain_auxiliary,a.selection_overrides)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
