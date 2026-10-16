#!/usr/bin/env python3
"""Auditable four-model runner. Configuration supplies benchmark-specific contracts.

Never upgrades native self-checks to direct cross-reference equivalence. Julia
kernel timing is valid only when structured records explicitly exclude JIT.
GPU use is serialized with the study-wide advisory lock.
"""
from __future__ import annotations
import argparse, csv, fcntl, hashlib, json, math, os, re, shutil, sqlite3
import statistics, subprocess, tempfile, struct
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get('STATE', ROOT / '.porting-state')) / 'portable_julia'
MODELS = ('cuda', 'cuda_jl', 'ka', 'jacc')
PROTOCOL = 'five_trials_last_four_v1'
EXPORT_MODEL_IDS = {'cuda':'cuda','cuda_jl':'julia','ka':'julia-ka','jacc':'julia-jacc'}

def discover(bench):
    base = ROOT / 'src' / f'{bench}-julia'
    return {'cuda': str(ROOT/'src'/f'{bench}-cuda'),
            'cuda_jl': str(base/f'{bench}-julia-cuda.jl') if (base/f'{bench}-julia-cuda.jl').exists() else str(base),
            'ka': str(base/f'{bench}-julia-KA'), 'jacc': str(base/f'{bench}-julia-JACC')}

def digest_tree(path):
    p = Path(path)
    if not p.exists(): return None
    h = hashlib.sha256()
    for f in sorted(p.rglob('*')):
        if f.is_file() and f.suffix in ('.jl','.cu','.cuh','.h','.cpp','.toml'):
            h.update(str(f.relative_to(p)).encode()); h.update(f.read_bytes())
    return h.hexdigest()

def run_command(command, cwd, env, timeout, log):
    try:
        p = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
        text = p.stdout + '\n' + p.stderr
        code = p.returncode
    except OSError as e:
        text=str(e); code=127
    except subprocess.TimeoutExpired as e:
        text = (e.stdout or b'').decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        text += '\nTIMEOUT'; code = 124
    log.write_text(text)
    return code, text

def finite_values(v):
    return isinstance(v,list) and bool(v) and all(isinstance(x,(float,int)) and math.isfinite(x) for x in v)

def artifact_values(row):
    artifact = row.get('output_artifact')
    if not artifact: return row.get('values')
    types = {'float32':'f','float64':'d','int32':'i','int64':'q','uint32':'I','uint64':'Q'}
    fmt = types.get(artifact.get('dtype'))
    if not fmt or artifact.get('byte_order','little') != 'little': return None
    data = Path(artifact['saved_path']).read_bytes()
    width=struct.calcsize('<'+fmt)
    if len(data) % width or len(data)//width != artifact.get('count'): return None
    return [x[0] for x in struct.iter_unpack('<'+fmt,data)]

def compare_records(ref, got, atol, rtol):
    a=ref.get('output_artifact'); b=got.get('output_artifact')
    if a and b and any(a.get(k)!=b.get(k) for k in ('dtype','count','shape')):
        return {'status':'mismatch','reason':'artifact dtype/count/shape mismatch'}
    return compare(artifact_values(ref),artifact_values(got),atol,rtol)

def save_artifacts(rows, cwd, out, model):
    for row in rows:
        artifact=row.get('output_artifact')
        if not artifact: continue
        path=Path(artifact['path'])
        if not path.is_absolute(): path=cwd/path
        saved=out/f"{model}_trial{row.get('trial')}_{path.name}"
        shutil.copyfile(path,saved)
        artifact['saved_path']=str(saved)
        artifact['sha256']=hashlib.sha256(saved.read_bytes()).hexdigest()

def compare(ref, got, atol, rtol):
    if not finite_values(ref) or not finite_values(got) or len(ref)!=len(got):
        return {'status':'unverified','reason':'missing/nonfinite/mismatched numerical values'}
    ae=[abs(a-b) for a,b in zip(ref,got)]
    re_=[e/max(abs(a),atol,1e-300) for a,e in zip(ref,ae)]
    ok=all(e<=atol+rtol*abs(a) for a,e in zip(ref,ae))
    return {'status':'pass' if ok else 'mismatch','checked_values':len(ae),'max_abs_error':max(ae),'max_rel_error':max(re_)}

def parse_records(text, spec, external_trial=None):
    rows=[]
    for line in text.splitlines():
        if line.startswith('STUDY_TRIAL '): rows.append(json.loads(line[len('STUDY_TRIAL '):]))
    if rows: return rows
    if external_trial is None: return []
    # Legacy reference adapters must declare the actual unit, timing region,
    # JIT handling and native verifier audit. No generic PASS inference.
    t=None
    if spec.get('time_regex'):
        m=re.search(spec['time_regex'],text)
        if m: t=float(m.group(1))*float(spec.get('time_scale_seconds',1))
    correct=None
    if spec.get('native_check_audited') and spec.get('pass_regex'):
        correct=bool(re.search(spec['pass_regex'],text)) and not bool(re.search(spec.get('fail_regex',r'(?i)\bFAIL(?:ED)?\b'),text))
    values=[]
    for pattern in spec.get('value_regexes',[]):
        for match in re.finditer(pattern,text): values.extend(float(x) for x in match.groups())
    return [{'trial':external_trial,'time_seconds':t,'correct':correct,'values':values,
             'checked_values':spec.get('checked_values'), 'jit_excluded':spec.get('jit_excluded',False),
             'timing_region':spec.get('timing_region'),'internal_repeats':spec.get('internal_repeats')}]

def export_global_csv():
    dbpath=STATE/'results.db'
    if not dbpath.exists(): return
    with (STATE/'export.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        db=sqlite3.connect(dbpath,timeout=60)
        records=[json.loads(x[0]) for x in db.execute('SELECT result_json FROM runs ORDER BY run_id')]
        db.close()
        target=STATE/'timings.csv'; tmp=STATE/'timings.csv.tmp'
        with tmp.open('w') as f:
            w=csv.writer(f)
            w.writerow(['run_id','benchmark','model','trial','time_seconds','correct','jit_excluded','mean_last_four_seconds','matched_performance_valid','reused_from_run_id'])
            for result in records:
                for model,data in result['models'].items():
                    for row in data.get('trials',[]):
                        w.writerow([result['run_id'],result['benchmark'],EXPORT_MODEL_IDS[model],row.get('trial'),row.get('time_seconds'),row.get('correct'),row.get('jit_excluded'),data.get('mean_seconds'),result.get('matched_performance_valid'),data.get('reused_from_run_id')])
        os.replace(tmp,target)
        alias=STATE/'verification.db'
        if not alias.exists(): alias.symlink_to('results.db')

def verify(config):
    bench=config['benchmark']; run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+bench
    out=STATE/'runs'/run_id; out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(config.get('gpu',6))
    paths=discover(bench); results={}; events=[]
    timeout=config.get('timeout',300)
    hardware=subprocess.run(['nvidia-smi','--query-gpu=index,name,uuid,compute_cap,driver_version','--format=csv,noheader'],capture_output=True,text=True).stdout
    (out/'hardware.txt').write_text(hardware)
    reuse_result=None
    reuse_config=None
    if config.get('reuse_result'):
        reuse_path=Path(config['reuse_result'])
        reuse_result=json.loads(reuse_path.read_text())
        reuse_config=json.loads((reuse_path.parent/'config.json').read_text())
        if reuse_result['benchmark']!=bench or reuse_result['protocol']!=PROTOCOL or reuse_result.get('input_contract')!=config.get('input_contract'):
            raise ValueError('reuse benchmark/protocol/input contract mismatch')
        for key in ('atol','rtol','gpu','configuration_id'):
            if reuse_config.get(key)!=config.get(key): raise ValueError('reuse config mismatch: '+key)
    for model in MODELS:
        spec=config.get('models',{}).get(model)
        if not spec:
            results[model]={'status':'missing','trials':[]}; continue
        cwd=Path(spec.get('cwd',paths[model])); cwd= cwd if cwd.is_absolute() else ROOT/cwd
        source_cwd=cwd
        adapter_file_hashes={}
        for arg in spec.get('command',[])+spec.get('build',[])+spec.get('source_dependencies',[]):
            f=Path(arg)
            if not f.is_absolute(): f=source_cwd/f
            if f.is_file() and f.suffix in ('.jl','.cu','.cuh','.h','.hpp','.cpp','.toml'):
                adapter_file_hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        source_hash=digest_tree(paths[model])
        if reuse_result and model in config.get('reuse_models',['cuda','cuda_jl','ka']):
            old=reuse_result['models'].get(model,{})
            if not old.get('complete') or old.get('source_hash')!=source_hash or reuse_config.get('models',{}).get(model)!=spec:
                raise ValueError('reuse source/config/incomplete mismatch: '+model)
            old_files=old.get('adapter_file_hashes')
            if old_files is not None and old_files!=adapter_file_hashes:
                raise ValueError('reuse adapter dependency changed: '+model)
            if old_files is None and not config.get('reuse_legacy_adapter_unchanged_verified'):
                raise ValueError('legacy bundle lacks individual adapter hashes: explicitly audit unchanged adapters first')
            results[model]=dict(old)
            results[model]['reused_from_run_id']=reuse_result['run_id']
            results[model]['adapter_file_hashes']=adapter_file_hashes
            events.append({'model':model,'phase':'reuse_verified_five_trial_bundle','source_run_id':reuse_result['run_id'],'legacy_adapter_audit':old_files is None})
            continue
        adapter_hash=digest_tree(source_cwd)
        temp=tempfile.TemporaryDirectory(prefix=f'hecbench-{model}-')
        cwd=Path(temp.name)
        for asset in spec.get('runtime_assets',[]):
            src=source_cwd/asset
            dst=cwd/asset
            dst.parent.mkdir(parents=True,exist_ok=True)
            if src.is_dir(): shutil.copytree(src,dst)
            else: shutil.copy2(src,dst)
        model_env=env.copy()
        model_env['STUDY_OUTPUT_DIR']=str(cwd)
        def resolve_args(args, building=False):
            resolved=[]
            for i,arg in enumerate(args):
                candidate=source_cwd/arg
                if i>0 and (i==0 or args[i-1] != '-o') and candidate.is_file() and candidate.suffix in ('.jl','.cu','.cpp','.c','.h','.cuh'):
                    resolved.append(str(candidate))
                elif i==0 and arg.startswith('./') and not spec.get('build'):
                    resolved.append(str(candidate))
                else: resolved.append(arg)
            return resolved
        build_ok=True
        if spec.get('build'):
            # ARCH supplied from hardware capability, never inherited sm120.
            query=subprocess.run(['nvidia-smi','-i',env['CUDA_VISIBLE_DEVICES'],'--query-gpu=compute_cap','--format=csv,noheader'],capture_output=True,text=True)
            arch='sm_'+query.stdout.strip().replace('.','')
            command=resolve_args([s.replace('{arch}',arch) for s in spec['build']],True)
            code,text=run_command(command,cwd,model_env,timeout,out/f'{model}_build.log')
            events.append({'model':model,'phase':'build','command':command,'returncode':code,'source_hash':source_hash})
            build_ok=code==0
        trials=[]
        if build_ok:
            count=1 if spec.get('structured',False) else 5
            with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                for n in range(1,count+1):
                    command=resolve_args(spec['command'])
                    code,text=run_command(command,cwd,model_env,timeout,out/f'{model}_run{n}.log')
                    events.append({'model':model,'phase':'execution_device_jit','command':spec['command'],'returncode':code,'source_hash':source_hash})
                    if code: break
                    try:
                        rows=parse_records(text,spec,None if count==1 else n)
                        save_artifacts(rows,cwd,out,model)
                        trials.extend(rows)
                    except (ValueError, OSError, KeyError) as error:
                        events.append({'model':model,'phase':'evidence_parse','error':str(error)})
                        break
        if temp: temp.cleanup()
        indices=[r.get('trial') for r in trials]
        complete=indices==[1,2,3,4,5]
        correct=complete and all(r.get('correct') is True for r in trials)
        times=[r.get('time_seconds') for r in trials]
        timing_valid=complete and all(isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in times) and all(r.get('jit_excluded') is True for r in trials)
        results[model]={'status':'native_check_pass' if correct else 'unverified','trials':trials,
                        'complete':complete,'timing_valid':timing_valid,'mean_seconds':statistics.mean(times[1:]) if timing_valid else None,
                        'source_hash':source_hash,'source_path':paths[model],'adapter_hash':adapter_hash,'adapter_source_path':str(source_cwd),'adapter_file_hashes':adapter_file_hashes,'export_model_id':EXPORT_MODEL_IDS[model], 'verification_method':spec.get('verification_method','unspecified'),
                        'native_check_description':spec.get('native_check_description'), 'limitations':spec.get('limitations',[])}
    comparisons={}
    for model in ('ka','jacc'):
        comparisons[model]={}
        for ref in ('cuda','cuda_jl'):
            a=results[ref].get('trials',[]); b=results[model].get('trials',[])
            checks=[compare_records(x,y,config.get('atol',1e-6),config.get('rtol',1e-5)) for x,y in zip(a,b)]
            if len(checks)==5 and all(x['status']=='pass' for x in checks):
                status='direct_artifact_pass' if all(x.get('output_artifact') and y.get('output_artifact') for x,y in zip(a,b)) else 'numeric_summary_pass'
            elif any(x['status']=='mismatch' for x in checks): status='mismatch'
            elif results[ref]['status']=='native_check_pass' and results[model]['status']=='native_check_pass': status='independent_native_checks_only'
            else: status='unverified'
            comparisons[model][ref]={'status':status,'trials':checks}
    ref_a=results['cuda'].get('trials',[])
    ref_b=results['cuda_jl'].get('trials',[])
    reference_checks=[compare_records(x,y,config.get('atol',1e-6),config.get('rtol',1e-5)) for x,y in zip(ref_a,ref_b)]
    reference_disagreement=any(x['status']=='mismatch' for x in reference_checks)
    reference_agreement=(len(reference_checks)==5 and all(x['status']=='pass' for x in reference_checks))
    variant_verified={}
    for model in ('ka','jacc'):
        comparisons_ok=all(v['status'] in ('direct_artifact_pass','numeric_summary_pass') for v in comparisons[model].values())
        trusted_native=(config.get('inputs_shared_verified') is True and
            all(results[m]['status']=='native_check_pass' and config.get('models',{}).get(m,{}).get('native_check_description') for m in ('cuda','cuda_jl',model)))
        variant_verified[model]=not reference_disagreement and results[model]['status']=='native_check_pass' and (comparisons_ok or trusted_native)
    regions={r.get('timing_region') for v in results.values() for r in v.get('trials',[])}
    matched=bool(config.get('inputs_shared_verified')) and bool(config.get('timed_regions_equivalent_verified')) and len(regions)==1 and None not in regions and all(v.get('timing_valid') and v.get('status')=='native_check_pass' for v in results.values()) and not reference_disagreement
    result={'run_id':run_id,'benchmark':bench,'configuration_id':config.get('configuration_id'),'tooling_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'database_path':str(STATE/'results.db'),'database_alias':str(STATE/'verification.db'),'protocol':PROTOCOL,'excluded_trial':1,'retained_trials':[2,3,4,5],
            'model_id_mapping':EXPORT_MODEL_IDS,'models':results,'comparisons':comparisons,'variant_verified':variant_verified,'reference_agreement':reference_agreement,'reference_disagreement':reference_disagreement,'reference_checks':reference_checks,'matched_performance_valid':matched,'input_contract':config.get('input_contract'),
            'ontology':config.get('ontology'), 'token_accounting':config.get('token_accounting',{'status':'unknown','input_tokens':None,'output_tokens':None}),
            'attempt_accounting':config.get('attempt_accounting'),'events':events}
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    db=sqlite3.connect(STATE/'results.db',timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,benchmark TEXT,result_json TEXT)')
    db.execute('INSERT INTO runs VALUES(?,?,?)',(run_id,bench,json.dumps(result,allow_nan=False))); db.commit(); db.close()
    with (out/'timings.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['run_id','benchmark','model','trial','time_seconds','correct','jit_excluded'])
        for model,v in results.items():
            for row in v.get('trials',[]): w.writerow([run_id,bench,EXPORT_MODEL_IDS[model],row.get('trial'),row.get('time_seconds'),row.get('correct'),row.get('jit_excluded')])
    export_global_csv()
    print(json.dumps({'result':str(out/'result.json'),'matched_performance_valid':matched,'comparisons':comparisons}))
    requested=[m for m in ('ka','jacc') if m in config.get('models',{})]
    return 0 if requested and all(variant_verified[m] for m in requested) else 2

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('benchmark',nargs='?'); p.add_argument('--config',type=Path); p.add_argument('--discover',action='store_true'); p.add_argument('--export',action='store_true')
    a=p.parse_args(); STATE.mkdir(parents=True,exist_ok=True)
    if a.export: export_global_csv(); return 0
    if a.discover: print(json.dumps(discover(a.benchmark),indent=2)); return 0
    if not a.config: p.error('--config required: benchmark-specific inputs/checker/timing contracts cannot safely be inferred')
    config=json.loads(a.config.read_text()); return verify(config)
if __name__=='__main__': raise SystemExit(main())
