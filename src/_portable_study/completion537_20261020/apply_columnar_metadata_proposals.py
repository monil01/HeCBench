#!/usr/bin/env python3
"""Root coordinator's reviewed, one-time join of worker-owned proposals.

The worker review denied shared mutation because root owns these paths. This
applies only eleven explicitly reviewed columnarSolver destinations, checks both
old and proposed hashes, and preserves every replaced byte before writing.
"""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'src/columnarSolver-julia'
OUT = Path(__file__).resolve().parent / 'columnar_metadata_join'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

def main():
    review = BASE / 'metadata_proposals/review.json'
    proposals = json.loads(review.read_text())['proposals']
    allowed = {
        '.porting-state/porting_logs/columnarSolver-julia/manifest.json',
        '.porting-state/porting_logs/columnarSolver-julia-ka/manifest.json',
        '.porting-state/porting_logs/columnarSolver-julia-jacc/manifest.json',
        '.porting-state/portable_julia/traits/columnarSolver.json',
        '.porting-state/portable_julia/scientific_intent_300/columnarSolver.json',
    }
    for model in ('cuda_jl', 'ka', 'jacc'):
        for folder in ('status_300', 'calibration_537'):
            allowed.add(f'.porting-state/portable_julia/{folder}/columnarSolver/{model}.json')
    assert len(proposals) == 11 and {e['destination'] for e in proposals} == allowed
    assert not (OUT / 'receipt.json').exists(), 'Already applied: inspect receipt rather than overwrite history.'
    for item in proposals:
        assert sha(ROOT / item['destination']) == item['expected_existing_sha256'], item
        assert sha(ROOT / item['proposal']) == item['proposal_sha256'], item
        json.loads((ROOT / item['proposal']).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for item in proposals:
        destination = ROOT / item['destination']
        if destination.exists():
            archive = OUT / 'before' / item['destination']
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, archive)
            assert sha(archive) == item['expected_existing_sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / item['proposal'], destination)
        assert sha(destination) == item['proposal_sha256']
    receipt = {
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Root-owned canonical join after explicit eleven-path/field review; no GPU, source or result changes.',
        'worker_boundary': 'Worker mutation was denied for ownership; reviewer explicitly allowed root-applied integration. Original worker proposals remain unchanged.',
        'review_sha256': sha(review),
        'proposals': proposals,
        'original_bytes_preserved': True,
        'historical_tokens': 'Original 186-token source proxy preserved, not promoted to historical actual usage.',
        'current_actuals': 'Closed prospective translation activity windows, context inclusive, never summed with shared/review windows.',
    }
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Applied eleven reviewed columnarSolver metadata proposals; originals preserved.')

if __name__ == '__main__':
    main()
