"""Population-specific labels for live reports; counts come from saved census."""
import json
import hashlib
from pathlib import Path

def context(root: Path, population: int):
    study=f'ontology/studies/julia_abstractions_{population}'; census_source=None
    if population==447:
        summary='The frozen 447-name assessment contains 436 qualified numerical contracts, two property contracts, five execution-only cases and four native hardware barriers.'
    else:
        path=root/'2026-PASC-Hecbench-Julia-paper/data/current_537_contract_census.json'
        if path.exists():
            d=json.loads(path.read_text());census_source={'path':str(path.relative_to(root)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'created_at':d.get('created_at'),'scope':'Selected saved all-four metadata census; not a fresh artifact or scientific correctness audit.'}
            if d.get('native_names')!=537:raise ValueError('537 report requires a 537-name census')
            counts=d['exclusive_outcome_counts'];hardware=0
            for benchmark in d['no_aggregate_eligible_accepted_contract_names']:
                statuses=[root/f'.porting-state/portable_julia/status_300/{benchmark}/{m}.json'for m in ('cuda_jl','ka','jacc')]
                if all(p.exists()and json.loads(p.read_text()).get('status')=='hardware_blocked'for p in statuses):hardware+=1
            remaining=537-d['accepted_named_contract_names']-hardware
            summary=(f"The 537-name catalogue has {d['accepted_named_contract_names']} accepted named contracts "
                     f"({counts.get('named_numerical',0)} numerical, {counts.get('property_equivalence',0)} property, "
                     f"{counts.get('execution_only',0)} execution-only), {hardware} native hardware barriers and "
                     f"{remaining} other names without an accepted all-four contract. "
                     'Named numerical contracts include explicit limited scopes and reference qualifications; they do not certify full applications.')
        else:
            summary='This view inventories the 537-name catalogue; a selected-contract census has not been loaded. Per-model outcomes remain individually qualified.'
        workflow=root/'ontology/studies/julia_abstractions_537/workflow_studies/word2vec/ka.json'
        if workflow.exists():summary+=' Word2vec also has a separately recorded full bounded asynchronous property workflow; distinct trained models are retained and no additional benchmark name is counted.'
    return {'population':population,'study_path':study,'summary':summary,'census_source':census_source}
