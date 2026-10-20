#!/usr/bin/env python3
"""Read-only saved-evidence integrity audit of one source/hash-bound 537 census.

No GPU runs, acceptance changes, or independent scientific-oracle claims. Checks
all primary output-artifact bytes and recorded trial/comparison gates. Auxiliary
raw buffers/property models are covered by the named benchmark-specific audits,
not silently included in this primary-artifact audit.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

WIDTHS = {'float32':4,'float64':8,'int8':1,'uint8':1,'int16':2,'uint16':2,'int32':4,'uint32':4,'int64':8,'uint64':8}
MODELS = ('cuda','cuda_jl','ka','jacc')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--census',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.source_root.resolve()
    census=json.loads(args.census.read_text());rows=[];issues=[];cache={};total_bytes=0;unique_bytes=0;counts=collections.Counter()
    for selected in census['rows']:
        bench=selected['benchmark'];errors=[];artifacts=[];p=root/selected['result']
        if sha(p)!=selected['result_sha256']:errors.append('selected_result_hash_mismatch')
        if sha(p.with_name('config.json'))!=selected['raw_config_sha256']:errors.append('selected_config_hash_mismatch')
        result=json.loads(p.read_text());execution=selected['outcome']=='execution_only'
        if result.get('reference_agreement') is not True or not all(result.get('variant_verified',{}).get(m) is True for m in ('ka','jacc')):errors.append('not_all_four_accepted')
        if result.get('protocol')!='five_trials_last_four_v1' or result.get('excluded_trial')!=1 or result.get('retained_trials')!=[2,3,4,5]:errors.append('trial_protocol_mismatch')
        for model in MODELS:
            record=result['models'][model];trials=record.get('trials',[])
            if len(trials)!=5 or [t.get('trial')for t in trials]!=[1,2,3,4,5]:errors.append(model+':five_trial_ids')
            times=[t.get('time_seconds')for t in trials]
            if not all(isinstance(t,(int,float)) and math.isfinite(t) and t>=0 for t in times):errors.append(model+':invalid_time')
            elif len(times)==5 and not math.isclose(sum(times[1:])/4,record.get('mean_seconds',-1),rel_tol=1e-12,abs_tol=1e-18):errors.append(model+':mean_not_last_four')
            if not record.get('complete') or not record.get('provenance_valid',True):errors.append(model+':incomplete_or_invalid_provenance')
            for trial in trials:
                counts['trial_rows']+=1
                if trial.get('correct') is not True:errors.append(model+':recorded_self_check_not_pass')
                artifact=trial.get('output_artifact')
                if execution:
                    if artifact or trial.get('checked_values')!=0:errors.append(model+':unexpected_execution_only_numerical_output')
                    continue
                if not artifact:errors.append(model+':missing_primary_artifact');continue
                f=Path(artifact['saved_path']);f=f if f.is_absolute()else root/f
                entry={'model':model,'trial':trial.get('trial'),'path':str(f),'dtype':artifact.get('dtype'),'count':artifact.get('count'),'expected_sha256':artifact.get('sha256')}
                try:
                    before=f.stat();width=WIDTHS.get(artifact.get('dtype'));expected=artifact.get('count',0)*(width or 0)
                    if width is None or artifact.get('byte_order','little')!='little' or before.st_size!=expected:errors.append(model+':artifact_format_or_size_mismatch')
                    key=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)
                    if key not in cache:cache[key]=sha(f);unique_bytes+=before.st_size
                    actual=cache[key];after=f.stat()
                    if (after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(before.st_size,before.st_mtime_ns,before.st_ctime_ns):errors.append(model+':artifact_changed_during_audit')
                    entry.update(actual_sha256=actual,size_bytes=before.st_size,integrity='pass'if actual==artifact.get('sha256')else'failed')
                    if actual!=artifact.get('sha256'):errors.append(model+':artifact_hash_mismatch')
                    total_bytes+=before.st_size;counts['artifact_paths']+=1
                except OSError as exc:entry['integrity']='missing_or_unreadable';errors.append(model+':'+str(exc))
                artifacts.append(entry)
        for model in ('ka','jacc'):
            for ref in ('cuda','cuda_jl'):
                comparison=result.get('comparisons',{}).get(model,{}).get(ref,{})
                expected='execution_only_pass'if execution else'direct_artifact_pass'
                if comparison.get('status')!=expected or len(comparison.get('trials',[]))!=5 or not all(t.get('status')==('execution_only_pass'if execution else'pass')for t in comparison.get('trials',[])):errors.append(model+':recorded_dual_reference_gate_'+ref)
                counts['recorded_comparison_trial_gates']+=len(comparison.get('trials',[]))
        errors=sorted(set(errors));issues.extend({'benchmark':bench,'issue':x}for x in errors)
        rows.append({'benchmark':bench,'result':selected['result'],'result_sha256':selected['result_sha256'],'outcome':selected['outcome'],'qualification':selected['qualification'],'issues':errors,'primary_artifacts':artifacts})
        print(json.dumps({'benchmark':bench,'issues':len(errors),'artifact_paths_so_far':counts['artifact_paths']}),flush=True)
    out={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'audit_source_sha256':sha(Path(__file__)),'census_path':str(args.census),'census_sha256':sha(args.census),'scope':'Full-byte integrity of selected primary saved output artifacts plus recomputed means and recorded all-five dual-reference gates. Does not rerun GPU kernels or scientific oracles, certify full CLI coverage, recheck auxiliary property-model buffers, or promote property/execution outcomes to numerical identity. Hardlink hash reuse only for same device/inode/size/mtime/ctime during this audit; every pathname stat verified.','summary':{'selected_benchmark_names':len(rows),'exclusive_outcomes':dict(collections.Counter(x['outcome']for x in rows)),**counts,'primary_artifact_bytes_across_paths':total_bytes,'unique_inodes_hashed':len(cache),'unique_primary_artifact_bytes_hashed':unique_bytes,'hard_issue_count':len(issues)},'issues':issues,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary']),flush=True)
    return 1 if issues else 0

if __name__=='__main__':raise SystemExit(main())
