#!/usr/bin/env python3
"""Continue the existing campaign after an owned controller's safe boundary.

Never interrupts a GPU process. The existing stop marker lets the old controller
finish its benchmark and freeze evidence; the replacement skips frozen names.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,required=True)
    ap.add_argument('--old-pid',type=int,required=True)
    args=ap.parse_args();root=args.source_root.resolve();raw=args.raw_root.resolve()
    proc=Path('/proc')/str(args.old_pid)
    if proc.stat().st_uid!=os.getuid() or b'v100_roofline_campaign.py'not in(proc/'cmdline').read_bytes():
        ap.error('Only the existing owned campaign controller is supported')
    start=(proc/'stat').read_text().split(')')[1].split()[19]
    stop=raw/'STOP_AFTER_BENCHMARK';expected=stop.read_bytes()
    if not expected.startswith(b'Root handoff to batched metadata freezing;'):
        ap.error('Recorded safe-boundary handoff marker required')
    journal=raw/'handoff_completion.jsonl'
    def note(**row):
        row['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with journal.open('a')as out:out.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    note(event='waiting_for_owned_controller_boundary',pid=os.getpid(),old_pid=args.old_pid,old_start=start)
    while True:
        try:
            fields=(proc/'stat').read_text().split(')')[1].split()
            if fields[0]=='Z' or fields[19]!=start:break
        except FileNotFoundError:break
        time.sleep(5)
    if stop.read_bytes()!=expected:raise SystemExit('Handoff marker changed; refusing duplicate controller')
    events=[json.loads(x)for x in(raw/'campaign_003.jsonl').read_text().splitlines()]
    if not any(x.get('event')=='controller_stopped_at_benchmark_boundary'for x in events):
        raise SystemExit('Old controller did not confirm safe boundary; preserve state for review')
    stop.unlink()
    command=[sys.executable,str(root/'tools/v100_roofline_campaign.py'),
        '--source-root',str(root),'--raw-root',str(raw),'--campaign-id','003',
        '--reviews',str(root/'ontology/studies/v100_roofline_537/approved_isolation_reviews_v7.json'),
        '--timeout','180','--asset-cap-mib','1024','--skip-frozen','--freeze-every','10',
        '--stop-file',str(stop)]
    for name in ['bfs','maxFlops','nbody','triad']:command+=['--exclude-benchmark',name]
    with(raw/'campaign_003_completion_stdout.jsonl').open('a')as log:
        child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with Path('/tmp/hecbench-roofline-metadata.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=root/'.porting-state/portable_julia/active_work_537.json';state=json.loads(path.read_text())
            state['roofline_campaign'].update(collector_pid=child.pid,collector_scope='Resumed003, isolationv7, every10names metadata freeze; raw evidence saved per model.',handoff_journal=str(journal),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            temporary=path.with_suffix('.json.next');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(path)
        note(event='replacement_controller_started',pid=child.pid,command=command)
        result=child.wait();note(event='replacement_controller_exited',returncode=result,
            meaning='Queue completion is not scientific completeness; inspect source-bound terminal assessments and failures.')
        raise SystemExit(result)


if __name__=='__main__':main()
