#!/usr/bin/env python3
"""Refresh auditable CPU exports while explicitly named owned collectors run.

No GPU execution, Git mutation, package installation or privilege operation.
Finishing a queue never marks the scientific campaign complete.
"""
import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--wait-pid',type=int,action='append',default=[])
    ap.add_argument('--interval',type=int,default=600)
    ap.add_argument('--once',action='store_true')
    ap.add_argument('--publication-stratum',help='Exact method/version/review SHA256; mandatory for multi-stratum canonical figures/publication predictions.')
    args=ap.parse_args();root=args.source_root.resolve();raw=args.raw_root.resolve()
    if not raw.is_relative_to('/tmp') or raw==Path('/tmp') or args.interval<60:
        ap.error('Owned raw root and interval >=60 seconds required')
    study=root/'ontology/studies/v100_roofline_537';journal=raw/'refresh.jsonl'
    tracked={}
    for pid in args.wait_pid:
        proc=Path('/proc')/str(pid)
        if not proc.exists():continue
        if proc.stat().st_uid!=os.getuid():ap.error('Only own collector PIDs may be monitored')
        command=(proc/'cmdline').read_bytes()
        if not any(name in command for name in [b'v100_roofline_campaign.py',b'v100_roofline_followups.py',b'v100_roofline_collect.py',b'v100_roofline_retry.py',b'v100_roofline_handoff.py']):
            ap.error('PID is not a roofline collector/controller')
        tracked[pid]=(proc/'stat').read_text().split(')')[1].split()[19]
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    def note(event):
        event['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with journal.open('a')as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    def run(label,argv):
        value=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True)
        note(dict(event=label,returncode=value.returncode,stdout=value.stdout[-4000:],stderr=value.stderr[-4000:]))
        return value.returncode==0
    def live():
        result=[]
        for pid,start in tracked.items():
            try:
                fields=(Path('/proc')/str(pid)/'stat').read_text().split(')')[1].split()
                if fields[0]!='Z' and fields[19]==start:result.append(pid)
            except (FileNotFoundError,ProcessLookupError):pass
        return result
    def refresh():
        if not run('summarize',[sys.executable,str(root/'tools/v100_roofline_summarize.py'),'--source-root',str(root),'--raw-root',str(raw)]):return
        if not run('integrity_audit',[sys.executable,str(root/'tools/v100_roofline_audit.py'),'--source-root',str(root),'--output',str(study/'integrity_audit.json')]):return
        # Mark both consumers unavailable before a fresh attempt; a crash cannot bless old files.
        with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for target in (study/'figures/canonical_figure_manifest.json',study/'publication_selection_status.json'):
                target.parent.mkdir(exist_ok=True)
                tmp=target.with_suffix('.json.next')
                tmp.write_text(json.dumps({'canonical_available':False,'current_export_available':False,'reason':'refresh_pending','requested_stratum':args.publication_stratum})+'\n');tmp.replace(target)
        # Plot in a fresh directory. Never fall back to a previous canonical PDF.
        # A manifest is the sole canonical authority; old files alone are not current evidence.
        with tempfile.TemporaryDirectory(prefix='roofline-figures-',dir='/tmp')as workspace:
            command=['/home/imo/noback/conda/bin/python3.9',str(root/'tools/v100_roofline_plot.py'),'--output-dir',workspace]
            if args.publication_stratum:command+=['--publication-stratum',args.publication_stratum]
            ok=run('roofline_plot',command)
            with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                dest=study/'figures';dest.mkdir(exist_ok=True)
                manifest={'canonical_available':False,'reason':'fresh_plot_failed_or_ambiguous','requested_stratum':args.publication_stratum}
                if ok:
                    meta=json.loads((Path(workspace)/'normalized_kernel_rooflines.metadata.json').read_text())
                    source_hash=hashlib.sha256((study/'kernel_roofline_points.csv').read_bytes()).hexdigest()
                    tensor=study/'kernel_tensor_roofline_points.csv'
                    tensor_hash=hashlib.sha256(tensor.read_bytes()).hexdigest()if tensor.is_file()else None
                    if (source_hash==meta['input']['sha256'] and tensor_hash==(meta['tensor_input']or{}).get('sha256') and meta['canonical_outputs']):
                        # Rebase paths in the metadata; retain all full raw identity strings.
                        for group in ('outputs','canonical_outputs'):
                            for item in meta[group]:item['path']=str(dest/Path(item['path']).name)
                        for item in meta['profiler_strata']:item['outputs']=[str(dest/Path(p).name)for p in item['outputs']]
                        for key in ('input','tensor_input'):
                            if meta.get(key):meta[key]['snapshot']=str(dest/Path(meta[key]['snapshot']).name)
                        for source in Path(workspace).iterdir():
                            if source.name.endswith('.metadata.json'):continue
                            target=dest/(source.name+'.next');shutil.copy2(source,target);target.replace(dest/source.name)
                        metadata=dest/'normalized_kernel_rooflines.metadata.json'
                        tmp=metadata.with_suffix('.json.next');tmp.write_text(json.dumps(meta,indent=2)+'\n');tmp.replace(metadata)
                        manifest={'canonical_available':True,'publication_selection':meta['publication_selection'],
                                  'metadata':{'path':str(metadata),'sha256':hashlib.sha256(metadata.read_bytes()).hexdigest()},
                                  'input_sha256':source_hash,'tensor_input_sha256':tensor_hash,'canonical_outputs':meta['canonical_outputs']}
                    else:manifest['reason']='input_changed_since_plot_or_no_compatible_points'
                manifest['created_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
                target=dest/'canonical_figure_manifest.json';tmp=target.with_suffix('.json.next');tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(target)

        output=root/'ontology/reports/ontology_report.html';temporary=output.with_suffix('.html.next')
        with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if run('ontology_report',[sys.executable,str(root/'ontology/tools/generate_ontology_report.py'),'--population','537','--output',str(temporary)]):
                temporary.replace(output)
        with tempfile.TemporaryDirectory(prefix='roofline-publication-',dir='/tmp')as workspace:
            export=Path(workspace)/'export'
            command=[sys.executable,str(root/'tools/export_v100_roofline_paper.py'),'--output-root',str(export)]
            if args.publication_stratum:command+=['--publication-stratum',args.publication_stratum]
            receipt=study/'publication_export_usage.json'
            if receipt.is_file():command+=['--usage-receipt',str(receipt)]
            publication_ok=run('publication_snapshot',command)
            # Invalidate publication authority on any failed/ambiguous fresh export; old files are history only.
            authority=study/'publication_selection_status.json'
            with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                value={'current_export_available':False,'requested_stratum':args.publication_stratum,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                if publication_ok:
                    snapshot=json.loads((export/'campaign_snapshot.json').read_text())
                    # Export read lock ended already: revalidate every bound input before publishing.
                    current=True
                    for name,binding in snapshot['input_bindings'].items():
                        path=Path(name);path=path if path.is_absolute()else root/path
                        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:current=False;break
                    if current:
                        dest=study/'publication';dest.mkdir(exist_ok=True);files={}
                        for source in export.iterdir():
                            if source.is_file():
                                next_path=dest/(source.name+'.next');shutil.copy2(source,next_path);next_path.replace(dest/source.name)
                                files[source.name]={'path':str(dest/source.name),'sha256':hashlib.sha256((dest/source.name).read_bytes()).hexdigest()}
                        value.update(current_export_available=True,publication_selection=snapshot['publication_selection'],files=files)
                    else:value['reason']='input_changed_since_export'
                temp=authority.with_suffix('.json.next');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(authority)
        with tempfile.TemporaryDirectory(prefix='roofline-retained-family-',dir='/tmp') as workspace:
            export=Path(workspace)/'export'
            if run('retained_family_aggregation',[sys.executable,str(root/'tools/export_v100_retained_roi.py'),'--source-root',str(root),'--output-root',str(export)]):
                with Path('/tmp/hecbench-roofline-metadata.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    dest=study/'retained_roi_aggregates';dest.mkdir(exist_ok=True)
                    for source in export.iterdir():
                        if source.is_file():
                            next_path=dest/(source.name+'.next');shutil.copy2(source,next_path);next_path.replace(dest/source.name)
        with tempfile.TemporaryDirectory(prefix='roofline-benchmark-assessment-',dir='/tmp') as workspace:
            export=Path(workspace)/'export'
            family=study/'retained_roi_aggregates/audit.json'
            command=[sys.executable,str(root/'tools/export_v100_benchmark_assessment.py'),
                     '--source-root',str(root),'--output-root',str(export)]
            if family.is_file():command+=['--retained-family-audit',str(family)]
            if run('benchmark_assessment',command):
                with Path('/tmp/hecbench-roofline-metadata.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    dest=study/'benchmark_assessment_snapshot';dest.mkdir(exist_ok=True)
                    manifest={}
                    for source in export.iterdir():
                        if not source.is_file():continue
                        packed=source.name in ('assessment.json','kernel_stage_observations.json')
                        name=source.name+('.gz' if packed else '')
                        temporary=dest/(name+'.next')
                        if packed:
                            with source.open('rb') as inp, temporary.open('wb') as raw_out:
                                with gzip.GzipFile(filename='',mode='wb',fileobj=raw_out,mtime=0) as zipped:
                                    shutil.copyfileobj(inp,zipped)
                        else:shutil.copy2(source,temporary)
                        data=temporary.read_bytes()
                        manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                                        'uncompressed_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                                        'uncompressed_bytes':source.stat().st_size}
                        temporary.replace(dest/name)
                    snapshot={'schema_version':1,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'files':manifest,'registry_assessment_complete':True,'GPU_ROI_campaign_complete':False,
                              'scope':'All537x4 statuses, selected-stage geometric projections and strictly admitted scientific families. No causal or complete elapsed ROI claims.'}
                    temporary=dest/'snapshot_manifest.json.next'
                    temporary.write_text(json.dumps(snapshot,indent=2)+'\n');temporary.replace(dest/'snapshot_manifest.json')
        run('joint_checkpoint',[sys.executable,str(root/'tools/checkpoint_portable_study.py'),'--population','537','--roofline-active'])
    note(dict(event='CPU_refresh_supervisor_started',pid=os.getpid(),tracked_process_start_times=tracked,
              meaning='Live derivative exports only. No automatic source/paper commits, claims of complete ROI or causal classification.'))
    previous=0
    while True:
        active=live()
        if time.monotonic()-previous>=args.interval or not active or args.once:
            refresh();previous=time.monotonic()
        if args.once or not active:
            note(dict(event='CPU_refresh_supervisor_finished',remaining_owned_collectors=active,
                      meaning='Queue end is not proof of complete scientific analysis; inspect failed/blocked/partial statuses.'))
            return
        time.sleep(15)


if __name__=='__main__':main()
