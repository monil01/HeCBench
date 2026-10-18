#!/usr/bin/env python3
"""Capture existing harness usage events; never launches a model or infers history."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'.porting-state/portable_julia'
p=argparse.ArgumentParser();p.add_argument('--agent',required=True);p.add_argument('--benchmark',required=True);p.add_argument('--model',choices=['shared','cuda_jl','ka','jacc'],required=True);p.add_argument('--phase',required=True);p.add_argument('--boundary',choices=['begin','end','abandon'],required=True);p.add_argument('--note',default='');a=p.parse_args()
agent=a.agent if a.agent.startswith('/root') else '/root/'+a.agent
source=json.loads((STATE/'usage_meter_sources.json').read_text())['agent_sources'][agent];path=Path(source['path'])
cohort_path=STATE/('batch_447.json' if (STATE/'batch_447.json').exists() else 'batch_300.json')
cohort=json.loads(cohort_path.read_text())['benchmarks']
if a.benchmark not in cohort+['study300','study150','study447'] or not a.phase.replace('-','').replace('_','').isalnum():raise SystemExit('Invalid frozen benchmark/phase path')
# Read only usage events from this explicitly registered agent stream.
with path.open('rb') as f:
 size=path.stat().st_size;f.seek(max(0,size-2**21));lines=f.read().decode(errors='replace').splitlines()
last=None
for line in lines:
 try:d=json.loads(line)
 except ValueError:continue
 x=d.get('payload',{})
 if x.get('type')=='token_count' and x.get('info',{}).get('total_token_usage'):last={'usage_event_timestamp':d.get('timestamp'),'cumulative_usage':x['info']['total_token_usage'],'source_token_event_sha256':hashlib.sha256(line.encode()).hexdigest()}
if last is None:raise SystemExit('No current usage event; do not infer zero or overwrite historical proxies.')
folder=STATE/'usage_boundaries_300'/a.benchmark;folder.mkdir(parents=True,exist_ok=True);out=folder/(a.model+'_'+a.phase+'.jsonl')
prior=[]
if out.exists():prior=[json.loads(x) for x in out.read_text().splitlines() if x.strip()]
open_starts=[x for x in prior if x['boundary']=='begin' and not any(e.get('begin_event_id')==x['event_id'] for e in prior if e['boundary'] in ['end','abandon'])]
if a.boundary=='begin' and open_starts:raise SystemExit('This phase already has an open boundary; close or adjudicate it explicitly.')
if a.boundary in ['end','abandon'] and len(open_starts)!=1:raise SystemExit('Need exactly one preceding begin; no historical backfill.')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();event={'event_id':hashlib.sha256((now+agent+a.benchmark+a.model+a.phase+a.boundary).encode()).hexdigest()[:24],'recorded_at':now,'agent':agent,'benchmark':a.benchmark,'implementation_model':a.model,'phase':a.phase,'boundary':a.boundary,'source_identity_sha256':source['source_identity_sha256'],'rollout_byte_size_at_read':size,**last,'note':a.note,'measurement_kind':'observed_harness_cumulative_usage_boundary','scope':'Consumed inference tokens during declared phase, including repeated/cached context and any coordination. Marker-turn alignment is approximate; not exclusive source translation or monetary billing. Cached input and reasoning are subsets, not extra totals. No historical recovery. Parent/shared workflow spans overlap subphases and must not be added to them.'}
event['marker_lag_seconds']=(datetime.datetime.fromisoformat(now)-datetime.datetime.fromisoformat(last['usage_event_timestamp'].replace('Z','+00:00'))).total_seconds()
event['accounting_level']='inclusive_workflow_overlaps_subphases' if a.phase=='benchmark_workflow' else 'declared_activity_window; interleaving must be disclosed'
event['meter_driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if a.boundary=='end':
 begin=open_starts[0]
 if begin['agent']!=agent or begin['source_identity_sha256']!=source['source_identity_sha256']:raise SystemExit('Usage stream changed across phase; cannot subtract.')
 delta={k:last['cumulative_usage'][k]-begin['cumulative_usage'][k] if k in begin['cumulative_usage'] else None for k in last['cumulative_usage']}
 if any(v is not None and v<0 for v in delta.values()):raise SystemExit('Counter reset detected; preserve begin and adjudicate before continuing.')
 event.update(begin_event_id=begin['event_id'],delta_usage=delta,elapsed_usage_event_start=begin['usage_event_timestamp'],positive_counter_span=delta.get('total_tokens') is not None and delta['total_tokens']>0)
if a.boundary=='abandon':
    if not a.note:raise SystemExit('Abandon requires an explicit measurement-gap reason.')
    event.update(begin_event_id=open_starts[0]['event_id'],delta_usage=None,measurement_status='incomplete_boundary_segment_unknown_cost')
with out.open('a') as f:f.write(json.dumps(event)+'\n')
print(json.dumps({'event_id':event['event_id'],'agent':agent,'benchmark':a.benchmark,'model':a.model,'phase':a.phase,'boundary':a.boundary,'usage_event_timestamp':last['usage_event_timestamp'],'delta_usage':event.get('delta_usage'),'path':str(out.relative_to(ROOT))}))
