#!/usr/bin/env python3
"""Run explicitly reviewed staged follow-up trees, with the shared GPU lock.

Each collector owns lock/idle/write/binding checks. Failures are retained, not
interpreted as translation failures or hidden by a retry in the same attempt.
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--plan-root',type=Path,action='append',required=True)
    ap.add_argument('--wait-controller',action='append',default=[],help='Wait for exact owned collector/controller PID:START_TICKS before starting this reviewed tree')
    ap.add_argument('--timeout',type=int,default=180)
    ap.add_argument('--exclude-benchmark',action='append',default=[])
    ap.add_argument('--skip-qualified-selected',action='store_true')
    ap.add_argument('--selection-overrides',type=Path,action='append',default=[])
    ap.add_argument('--large-artifacts',action='store_true',help='Only reviewed Perlin/ZMDDFT/SU3 roots; collector enforces bounded policy')
    ap.add_argument('--journal',type=Path,required=True)
    a=ap.parse_args();root=a.source_root.resolve();raw=a.raw_root.resolve()
    if not raw.is_relative_to('/tmp') or raw==Path('/tmp'):
        ap.error('Owned raw tree strictly under /tmp required')
    if not a.journal.resolve().is_relative_to(raw):ap.error('Journal must be in raw tree')
    for planroot in a.plan_root:
        if not planroot.resolve().is_relative_to(raw) or not planroot.name.startswith('batch'):
            ap.error('Explicit batch tree in raw root required')
    def note(event):
        event['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with a.journal.open('a')as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    selections={}
    for path in a.selection_overrides:
        benchmark=json.loads(path.read_text())['benchmark']
        if benchmark in selections:ap.error('One explicit selector per benchmark in this traversal')
        selections[benchmark]=path.resolve()
    for identity in a.wait_controller:
        pid_text,start=identity.split(':',1);pid=int(pid_text);proc=Path('/proc')/str(pid)
        if proc.exists():
            if proc.stat().st_uid!=os.getuid():ap.error('Only own dependency process may be monitored')
            argv=(proc/'cmdline').read_bytes().split(b'\0')
            permitted={b'v100_roofline_collect.py',b'v100_roofline_followups.py',b'v100_roofline_campaign.py',b'v100_roofline_retry.py'}
            if not any(x.rsplit(b'/',1)[-1] in permitted for x in argv):ap.error('Dependency is not an owned roofline collector/controller')
            if(proc/'stat').read_text().split(')')[1].split()[19]!=start:ap.error('Dependency PID start time changed')
        note(dict(event='waiting_for_owned_dependency',pid=pid,start_ticks=start))
        while proc.exists():
            try:
                fields=(proc/'stat').read_text().split(')')[1].split()
                if fields[0]=='Z' or fields[19]!=start:break
            except FileNotFoundError:break
            time.sleep(5)
        note(dict(event='owned_dependency_finished',pid=pid,start_ticks=start))
    for planroot in a.plan_root:
        command=[sys.executable,str(root/'tools/v100_roofline_collect.py'),
            '--source-root',str(root),'--plan-root',str(planroot),'--timeout',str(a.timeout)]
        benchmarks={json.loads(p.read_text())['benchmark'] for p in planroot.glob('*/*/plan.json')}
        if a.large_artifacts:command+=['--large-artifacts']
        for excluded in a.exclude_benchmark:command+=['--exclude-benchmark',excluded]
        if a.skip_qualified_selected:command+=['--skip-qualified-selected']
        if len(benchmarks)==1 and next(iter(benchmarks)) in selections:
            command+=['--selection-overrides',str(selections[next(iter(benchmarks))])]
        result=subprocess.run(command,
            cwd=root,capture_output=True,text=True)
        note(dict(event='followup_collect',plan_root=str(planroot),returncode=result.returncode,
                  stdout=result.stdout,stderr=result.stderr))
        summary=subprocess.run([sys.executable,str(root/'tools/v100_roofline_summarize.py'),
            '--source-root',str(root),'--raw-root',str(raw)],cwd=root,capture_output=True,text=True)
        note(dict(event='followup_freeze',plan_root=str(planroot),returncode=summary.returncode,
                  stdout=summary.stdout,stderr=summary.stderr))
        if summary.returncode:
            raise SystemExit(summary.returncode)
    note(dict(event='reviewed_followup_traversal_finished',
              meaning='Inspect per-case status, ROI coverage and missing counters; queue traversal is not complete scientific coverage.'))


if __name__=='__main__':main()
