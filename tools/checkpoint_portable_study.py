#!/usr/bin/env python3
"""Checkpoint translation and manuscript together; research stays local."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from study_evidence import effective_config

parser = argparse.ArgumentParser()
parser.add_argument('--complete', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
state = root / '.porting-state'
paper = root / '2026-PASC-Hecbench-Julia-paper'
local = state / 'portable_julia'

def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def git(path, *argv):
    return subprocess.check_output(['git', *argv], cwd=path, text=True).strip()

cohort_path = local / 'batch_300.json'
if not cohort_path.exists():
    cohort_path = local / 'batch_100.json'
cohort = read(cohort_path)
target = cohort['target_benchmarks']
selected = set(cohort['benchmarks'])
adjudications = read(paper / 'research/data/adjudications.json', {})
dispositions = read(paper / 'research/data/configuration_dispositions.json', {})
records = {}
for path in sorted((local / 'runs').glob('*/result.json')):
    d = read(path)
    if d['benchmark'] not in selected or adjudications.get(d['run_id'], {}).get('exclude_from_performance'):
        continue
    if not any(d.get('variant_verified', {}).values()):
        continue
    config = effective_config(path,read(path.with_name('config.json')))
    cid = d.get('configuration_id') or config['configuration_id']
    records[(d['benchmark'], cid)] = {
        'benchmark': d['benchmark'], 'configuration_id': cid,
        'run_id': d['run_id'], 'variants': d['variant_verified'],
        'matched_performance_valid': d.get('matched_performance_valid', False),
        'native_reference_repaired': bool(config.get('native_reference_repaired')),
        'native_behavior_known_scientific_defect': bool(config.get('native_behavior_known_scientific_defect')),
        'configuration_classification_adjudication': config.get('configuration_classification_adjudication'),
        'hardware_stratum': 'single_v100' if str(config.get('gpu',6))=='6' else 'multiple_or_other_devices',
        'untouched_single_gpu_aggregate_eligible': str(config.get('gpu',6))=='6' and not config.get('native_reference_repaired') and not config.get('exclude_from_aggregate',False) and not config.get('exclude_untouched_native_aggregates',False) and not config.get('exclude_from_scientific_fidelity_aggregates',False) and not config.get('native_behavior_known_scientific_defect',False),
        'role': dispositions.get(cid, {}).get('role', 'primary'),
        'references_agree': d.get('reference_agreement'),
        'result': str(path.relative_to(root)),
        'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'models': {k: {'source_hash': m.get('source_hash'),
                       'trials_seconds': [t.get('time_seconds') for t in m.get('trials', [])],
                       'trials_correct': [t.get('correct') for t in m.get('trials', [])],
                       'mean_last_four_seconds': m.get('mean_seconds')}
                   for k, m in d['models'].items()}}
primary = [r for r in records.values() if r['role'] == 'primary']
pairs = len({r['benchmark'] for r in primary if r['matched_performance_valid']})
versions = {k: len({r['benchmark'] for r in primary if r['variants'].get(k)})
            for k in ['ka', 'jacc']}
strata = {
    'known_native_scientific_defect_names': len({r['benchmark'] for r in primary if r['matched_performance_valid'] and r['native_behavior_known_scientific_defect']}),
    'corrected_native_reference_names': len({r['benchmark'] for r in primary if r['matched_performance_valid'] and r['native_reference_repaired']}),
    'multiple_or_other_device_names': len({r['benchmark'] for r in primary if r['matched_performance_valid'] and r['hardware_stratum']!='single_v100'}),
    'untouched_native_single_gpu_names': len({r['benchmark'] for r in primary if r['matched_performance_valid'] and r['untouched_single_gpu_aggregate_eligible']})}
heads = {k: git(p, 'rev-parse', 'HEAD') for k, p in [('source', root), ('paper', paper)]}
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S%fZ')
status = f'active_{target}_benchmark_expansion'
if args.complete:
    coverage = read(paper / 'research/data/coverage.json')
    audit = read(paper / 'research/data/numerical_contract_audit.json')
    assert pairs == target and coverage['matched_benchmarks'] == target
    assert audit['summary']['hard_issue_count'] == 0 and (paper / 'build/main.pdf').is_file()
    status = f'complete_{target}_pairs_manuscript_review_ready'
payload = {
    'created_at': now.isoformat(), 'status': status, 'heads': heads,
    'instruction_hashes': {n: hashlib.sha256((root/n).read_bytes()).hexdigest()
                           for n in ['AGENT_INSTRUCTIONS_CODEX_Julia.md', 'JULIA_CORRECTNESS_CHECK.md', 'paper_writing.md']},
    'cohort': str(cohort_path.relative_to(root)), 'target_benchmarks': target,
    'frozen_baseline': str((local/'baseline_300.json').relative_to(root)),
    'verified_matched_benchmarks': pairs, 'verified_portable_versions': versions,
    'verification_strata': strata, 'matched_count_includes_conditional_and_multigpu': True,
    'ownership': cohort.get('ownership', {}), 'all_configuration_evidence': list(records.values()),
    'model_status_inventory': str((local/'study_300_status.json').relative_to(root)),
    'dirty_files': {k: git(p, 'status', '--short') for k,p in [('source',root),('paper',paper)]},
    'next_translation': 'Continue disjoint frozen300 queues KA then JACC, repairing references where needed. Root commits verified variants separately.',
    'next_paper': 'Review scientific domains and per-model ontology/failures, update honest current counts and domain-scientist RQs. Build publication files when ready.',
    'local_only_research': True,
    'paper_build_artifact': {'path':str((paper/'build/main.pdf').relative_to(root)),
        'sha256':hashlib.sha256((paper/'build/main.pdf').read_bytes()).hexdigest() if (paper/'build/main.pdf').exists() else None,
        'sources_newer_than_pdf':[str(p.relative_to(paper)) for p in [paper/'main.tex',paper/'references.bib',*paper.glob('sections/**/*.tex'),*paper.glob('figures/*.tex'),*paper.glob('figures/*.pdf')] if p.exists() and (not (paper/'build/main.pdf').exists() or p.stat().st_mtime>(paper/'build/main.pdf').stat().st_mtime)],
        'scope':'File hash and source freshness, not a submission or full build-quality certification.'},
    'resume_notes': [
        'Read the current authority at the TOP of all three guides, then this checkpoint. Completed100 stop notes are superseded.',
        'Preserve unrelated deleted AGENT_INSTRUCTIONS_CODEX.md and md5hash caches. Root alone stages source commits.',
        'GPU6 runs are serialized by /tmp/hecbench-julia-study-gpu.lock; discover active worker/model logs before launching.',
        'Five trials, all correctness checks retained; exclude trial1 time and average trials2–5. No preferred reruns.',
        'Historic300 PASS labels are not strengthened verification. Unknown costs and unreviewed scientific annotations stay unknown.',
        'Research data/scripts/checkpoints stay local and ignored. Paper GitHub/Overleaf contains publication files only.',
        'Interpret implementation models cuda_jl/ka/jacc separately from Julia language/package release identifiers.']}
folder = paper/'research/checkpoints'
folder.mkdir(parents=True, exist_ok=True)
out = folder/('checkpoint_'+stamp+'.md')
out.write_text('# Translation and scientific-paper checkpoint\n\n'
               f'{pairs}/{target} matched benchmark pairs. Status: {status}.\n\n'
               'This expansion supersedes historical complete100 checkpoints. No new success is implied by a target count.\n\n```json\n'
               + json.dumps(payload, indent=2)+'\n```\n')
latest = {'checkpoint_id': stamp, 'path': str(out.relative_to(root)),
          'created_at': now.isoformat(), 'status': status, 'verified_pairs': pairs,
          'verified_portable_versions': versions, 'target_pairs': target,
          'verification_strata': strata, 'matched_count_includes_conditional_and_multigpu': True,
          'performance_protocol': 'five_trials_last_four_v1',
          'source_head': heads['source'], 'paper_head': heads['paper']}
tmp = folder/'LATEST.json.tmp'
tmp.write_text(json.dumps(latest, indent=2)+'\n')
tmp.replace(folder/'LATEST.json')
# A small tracked-source pointer can be committed without shipping research logs.
summary = root/'ontology/studies/julia_abstractions_300/PROGRESS.json'
summary.parent.mkdir(parents=True, exist_ok=True)
summary.write_text(json.dumps(latest, indent=2)+'\n')
print(json.dumps(latest))
