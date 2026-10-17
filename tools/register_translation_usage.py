#!/usr/bin/env python3
"""Register only this root thread and its direct worker usage streams."""
import datetime,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'.porting-state/portable_julia';thread=os.environ.get('CODEX_THREAD_ID')
if not thread:raise SystemExit('Current root thread identity unavailable; no arbitrary session log selection.')
mapping={}
for p in (Path.home()/'.codex/sessions').rglob('*.jsonl'):
 try:
  with p.open() as f:m=json.loads(f.readline()).get('payload',{})
 except (OSError,ValueError):continue
 source=m.get('source',{});spawn=source.get('subagent',{}).get('thread_spawn',{}) if isinstance(source,dict) else {}
 agent='/root' if thread in p.name else spawn.get('agent_path') if spawn.get('parent_thread_id')==thread else None
 if agent in ['/root','/root/maps_ports','/root/stencil_ports','/root/portable_infrastructure']:mapping[agent]={'path':str(p),'source_identity_sha256':hashlib.sha256(p.name.encode()).hexdigest()}
if '/root' not in mapping:raise SystemExit('Root rollout not found; cannot register this study reliably.')
p=STATE/'usage_meter_sources.json';now=datetime.datetime.now(datetime.timezone.utc).isoformat()
if p.exists():
 with p.with_name('usage_meter_sources_history.jsonl').open('a') as f:f.write(json.dumps({'superseded_at':now,'mapping':json.loads(p.read_text()),'scope':'Retain prior source identities; never subtract across streams.'})+'\n')
d={'registered_at':now,'agent_sources':mapping,'scope':'Only this study root and its direct namedworkers. Read usage events only; no prompts/messages/personal metadata exported. New thread identities require new segments; abandon old open markers with unknown cost rather than infer deltas.'};tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p);print(json.dumps({'registered_study_streams':sorted(mapping),'path':str(p.relative_to(ROOT))}))
