#!/usr/bin/env python3
"""Retry only selected externally stopped or timed-out profiling processes.

Fresh plans preserve inputs, mathematics and whole-program checks. Timeouts use
the previously validated first-specialization selector across launch geometries;
this explicitly reduces geometric coverage, never certifies an entire ROI.
"""
import argparse
import datetime
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--batch-id', required=True)
    ap.add_argument('--reviews', type=Path, required=True)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--freeze-every', type=int, default=10)
    ap.add_argument('--timeout', type=int, default=600)
    args = ap.parse_args()
    root, raw = args.source_root.resolve(), args.raw_root.resolve()
    if not raw.is_relative_to('/tmp') or raw == Path('/tmp'):
        ap.error('Owned raw tree under /tmp required')
    if not args.batch_id.isalnum() or args.freeze_every < 1:
        ap.error('Alphanumeric fresh batch ID and positive freeze interval required')
    study = root/'ontology/studies/v100_roofline_537'
    journal = raw/('retry_'+args.batch_id+'.jsonl')
    def note(**value):
        value['time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with journal.open('a') as output:
            output.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    frontier = {}
    with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        for path in sorted((study/'benchmarks').glob('*.json')):
            ledger = json.loads(path.read_bytes())
            for model, state in ledger['models'].items():
                if state['profiling_status'] != 'profile_process_failed':
                    continue
                ids = state.get('profile_ids', [])
                if not ids:
                    continue
                provenance_path = root/ids[0]/'profile_provenance.json.gz'
                if not provenance_path.is_file():
                    continue
                packed = provenance_path.read_bytes()
                provenance = json.loads(gzip.decompress(packed))
                reason = provenance.get('profile', {}).get('stop_reason')
                if reason not in ('external_gpu_activity_detected_own_profile_stopped', 'wall_timeout'):
                    continue
                frontier.setdefault(ledger['benchmark'], []).append(dict(
                    model=model, reason=reason, configuration_id=ledger['accepted_named_contract']['configuration_id'],
                    parent_provenance=str(provenance_path), parent_compressed_sha256=sha(packed)))
    frontier_path = raw/('retry_'+args.batch_id+'_frontier.json')
    if frontier_path.exists():
        ap.error('Fresh batch ID required; prior frontier and attempts are immutable')
    frontier_path.write_text(json.dumps(frontier, indent=2)+'\n')
    note(event='retry_frontier', names=len(frontier), slots=sum(map(len, frontier.values())),
         frontier=str(frontier_path), frontier_sha256=sha(frontier_path.read_bytes()),
         meaning='Shared measurement recovery only. Old failures retained; no translation or timing changes.')
    if not args.execute:
        return
    pending = 0
    def freeze():
        result = subprocess.run([sys.executable, str(root/'tools/v100_roofline_summarize.py'),
            '--source-root', str(root), '--raw-root', str(raw)], cwd=root, capture_output=True, text=True)
        note(event='retry_freeze', returncode=result.returncode,
             stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
    for benchmark, entries in frontier.items():
        # Refresh may have selected a better result while this frontier waited.
        with Path('/tmp/hecbench-roofline-metadata.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            state = json.loads((study/'benchmarks'/(benchmark+'.json')).read_bytes())['models']
            entries = [x for x in entries if state[x['model']]['profiling_status'] != 'profile_process_completed']
        if not entries:
            continue
        plans = raw/('batch'+args.batch_id+'-retry-'+benchmark)
        prepare = [sys.executable, str(root/'tools/v100_roofline_prepare.py'),
            '--source-root', str(root), '--benchmark', benchmark,
            '--models', *[x['model'] for x in entries], '--output-root', str(plans),
            '--reviews', str(args.reviews.resolve()), '--asset-cap-mib', '1024', '--stage']
        result = subprocess.run(prepare, cwd=root, capture_output=True, text=True)
        note(event='retry_prepare', benchmark=benchmark, returncode=result.returncode,
             stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            continue
        command = [sys.executable, str(root/'tools/v100_roofline_collect.py'),
            '--source-root', str(root), '--plan-root', str(plans), '--timeout', str(args.timeout)]
        if any(x['reason'] == 'wall_timeout' for x in entries):
            selector = plans/'bounded_selector.json'
            receipt = dict(benchmark=benchmark, configuration_id=entries[0]['configuration_id'],
                fixture_description=dict(path=str(frontier_path), sha256=sha(frontier_path.read_bytes())),
                expected_matching_launches=None,
                selection={x['model']:['--filter-mode','global','--kernel-name-base','mangled',
                    '--kernel-id','::regex:.*:1'] for x in entries},
                coverage='First invocation per mangled specialization across geometries; geometry/state/phase undercoverage. '
                    'Existing selector canary validated global versus per-launch-config; no launch-count extrapolation.',
                parent_failures=entries)
            selector.write_text(json.dumps(receipt, indent=2)+'\n')
            command += ['--selection-overrides', str(selector)]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True)
        note(event='retry_collect', benchmark=benchmark, returncode=result.returncode,
             stdout=result.stdout, stderr=result.stderr)
        pending += 1
        if pending >= args.freeze_every:
            freeze()
            pending = 0
    if pending:
        freeze()
    note(event='retry_traversal_finished', meaning='Inspect remaining statuses. Queue traversal is not complete scientific coverage.')


if __name__ == '__main__':
    main()
