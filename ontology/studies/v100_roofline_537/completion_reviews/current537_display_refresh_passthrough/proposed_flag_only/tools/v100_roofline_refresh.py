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
    ap.add_argument('--current537-display-review-sha256',help='Pass exact root-approved current537 display authority SHA to report generation only.')
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
        run('roofline_plot',['/home/imo/noback/conda/bin/python3.9',str(root/'tools/v100_roofline_plot.py')])
        output=root/'ontology/reports/ontology_report.html';temporary=output.with_suffix('.html.next')
        with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            report_command=[sys.executable,str(root/'ontology/tools/generate_ontology_report.py'),'--population','537','--output',str(temporary)]
            if args.current537_display_review_sha256:report_command+=['--current537-display-review-sha256',args.current537_display_review_sha256]
            if run('ontology_report',report_command):
                temporary.replace(output)
        with tempfile.TemporaryDirectory(prefix='roofline-publication-',dir='/tmp')as workspace:
            export=Path(workspace)/'export'
            command=[sys.executable,str(root/'tools/export_v100_roofline_paper.py'),'--output-root',str(export)]
            receipt=study/'publication_export_usage.json'
            if receipt.is_file():command+=['--usage-receipt',str(receipt)]
            if run('publication_snapshot',command):
                dest=study/'publication';dest.mkdir(exist_ok=True)
                for source in export.iterdir():
                    if source.is_file():
                        next_path=dest/(source.name+'.next');shutil.copy2(source,next_path);next_path.replace(dest/source.name)
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
