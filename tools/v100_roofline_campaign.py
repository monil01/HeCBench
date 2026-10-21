#!/usr/bin/env python3
"""Serial resumable V100 profiling campaign; stages one benchmark at a time.

Fresh scratch bounds the working set on disk. No existing profiler attempt is
overwritten. The GPU collector owns device/lock/idle guards for each execution.
"""
import argparse
import datetime
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
    ap.add_argument('--campaign-id',required=True)
    ap.add_argument('--benchmark',action='append')
    ap.add_argument('--exclude-benchmark',action='append',default=[])
    ap.add_argument('--reviews',type=Path)
    ap.add_argument('--selection-overrides',type=Path)
    ap.add_argument('--skip-qualified-selected',action='store_true',help='Preserve successful selected captures already supplied by reviewed follow-ups')
    ap.add_argument('--timeout',type=int,default=180)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--freeze-every',type=int,default=1,help='Freeze metadata after this many names; raw per-model evidence is saved immediately')
    ap.add_argument('--skip-frozen',action='store_true',
                    help='Resume queue after benchmark freeze events already journaled; failures still need separate explicit follow-up attempts.')
    ap.add_argument('--stop-file',type=Path,
                    help='If this file exists after a benchmark freeze, stop the controller at that boundary without interrupting a GPU child.')
    ap.add_argument('--asset-cap-mib',type=int,default=512)
    a=ap.parse_args();root=a.source_root.resolve();raw=a.raw_root.resolve()
    if a.freeze_every<1:ap.error('freeze-every must be positive')
    raw.relative_to(Path('/tmp'))
    if not a.campaign_id.replace('-','').replace('_','').isalnum():
        ap.error('Use an alphanumeric campaign ID with optional hyphens/underscores')
    census=root/'src/_portable_study/completion537_20261020/current537_census.json'
    population=json.loads(census.read_text())
    names=sorted(x['benchmark']for x in population['rows'])
    if a.benchmark:names=[b for b in names if b in a.benchmark]
    names=[b for b in names if b not in a.exclude_benchmark]
    if a.limit:names=names[:a.limit]
    journal=raw/('campaign_'+a.campaign_id+'.jsonl')
    already_frozen=set()
    if a.skip_frozen and journal.exists():
        for line in journal.read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('event')=='frozen':already_frozen.add(event['benchmark'])
    def note(event):
        event.update(time=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with journal.open('a')as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    note({'event':'start_or_resume','pid':os.getpid(),'benchmarks':len(names),
          'campaign_id':a.campaign_id,'root':str(root)})
    unflushed=[]
    for index,b in enumerate(names,1):
        if b in already_frozen:
            continue
        batch=raw/('batch'+a.campaign_id+'-'+b)
        if not batch.exists():
            prepare=[sys.executable,str(root/'tools/v100_roofline_prepare.py'),
                '--source-root',str(root),'--census',str(census),'--output-root',str(batch),
                '--benchmark',b,'--stage','--asset-cap-mib',str(a.asset_cap_mib)]
            if a.reviews:prepare+=['--reviews',str(a.reviews)]
            result=subprocess.run(prepare,cwd=root,capture_output=True,text=True)
            note({'event':'prepare','benchmark':b,'ordinal':index,'returncode':result.returncode,
                  'stdout':result.stdout,'stderr':result.stderr})
            if result.returncode:
                # Preserve incomplete staging; repair/review explicitly on resume.
                continue
        collector=[sys.executable,str(root/'tools/v100_roofline_collect.py'),
            '--source-root',str(root),'--plan-root',str(batch),'--timeout',str(a.timeout)]
        if a.skip_qualified_selected:collector+=['--skip-qualified-selected']
        if a.selection_overrides:collector+=['--selection-overrides',str(a.selection_overrides)]
        result=subprocess.run(collector,cwd=root,capture_output=True,text=True)
        note({'event':'collect','benchmark':b,'ordinal':index,'returncode':result.returncode,
              'stdout':result.stdout,'stderr':result.stderr})
        unflushed.append(b)
        if len(unflushed)<a.freeze_every and index<len(names) and not(a.stop_file and a.stop_file.exists()):
            note({'event':'raw_saved_metadata_freeze_deferred','benchmark':b,'unflushed_names':list(unflushed)})
            continue
        # Root alone updates ontology measurement records. A single campaign
        # controller must run; never start concurrent controllers on this tree.
        summarize=[sys.executable,str(root/'tools/v100_roofline_summarize.py'),
                   '--source-root',str(root),'--raw-root',str(raw)]
        result=subprocess.run(summarize,cwd=root,capture_output=True,text=True)
        if result.returncode:
            note({'event':'freeze_failed_stop','benchmark':b,'stderr':result.stderr})
            raise SystemExit(result.returncode)
        for frozen_name in unflushed:
            note({'event':'frozen','benchmark':frozen_name,'ordinal':index,'summary':json.loads(result.stdout) if frozen_name==b else {'scope':'Part of same complete metadata refresh','last_benchmark':b}})
        unflushed=[]
        if a.stop_file and a.stop_file.exists():
            note({'event':'controller_stopped_at_benchmark_boundary','benchmark':b,
                  'meaning':'Explicit controller handoff; scientific task remains active.'})
            return
    if unflushed:
        result=subprocess.run([sys.executable,str(root/'tools/v100_roofline_summarize.py'),
            '--source-root',str(root),'--raw-root',str(raw)],cwd=root,capture_output=True,text=True)
        if result.returncode:
            note({'event':'freeze_failed_stop','benchmarks':unflushed,'stderr':result.stderr})
            raise SystemExit(result.returncode)
        for frozen_name in unflushed:
            note({'event':'frozen','benchmark':frozen_name,'summary':{'scope':'Final complete metadata refresh'}})
    note({'event':'selected_plan_traversal_finished','benchmarks':len(names),
          'meaning':'Traversal is not complete roofline/ROI/bottleneck analysis; inspect per-case statuses.'})


if __name__=='__main__':main()
