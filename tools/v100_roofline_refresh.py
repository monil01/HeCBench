#!/usr/bin/env python3
"""Refresh auditable CPU exports while explicitly named owned collectors run.

No GPU execution, Git mutation, package installation or privilege operation.
Finishing a queue never marks the scientific campaign complete.
"""
import argparse
import datetime
import fcntl
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
        if not any(name in command for name in [b'v100_roofline_campaign.py',b'v100_roofline_followups.py',b'v100_roofline_collect.py']):
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
            if run('ontology_report',[sys.executable,str(root/'ontology/tools/generate_ontology_report.py'),'--population','537','--output',str(temporary)]):
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
